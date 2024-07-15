import numpy as np

np.seterr(divide="ignore", invalid="ignore")

import matplotlib.pyplot as plt
import time
import cProfile


"""
Code to simulate the growth of a population of cells using the Gillespie algorithm.
To plot trustfull data, the code uses multiple simulations and averages the results.
"""

MAX_POPULATION = 1000
N_SIMULATIONS = 5
MAX_NAME = 1

# Initialize parameters

np.random.seed(0)
initial_population = 100
total_time = 200
death_rate = 0.01
initial_growth_rate = 0.01
std_growth_rate = 0.00005
std_growth_rate_list = [std_growth_rate]
get_proportion_per_name_list = False
dt = 0.5
chemostat = (6600, 0.7)
variation_rate = 0.1
smooth_coefficient = 1000


def general_logistic_growth(
    population, growth_rate, carrying_capacity, alpha=1, beta=0, gamma=0
):
    return (
        growth_rate
        * (population**alpha)
        * (1 - (population / carrying_capacity) ** beta) ** gamma
    )


def exponential_growth(
    population, growth_rate, carrying_capacity, alpha=1, beta=0, gamma=0
):
    return general_logistic_growth(population, growth_rate, np.inf, 1, 0, 0)


def logistic_growth(
    population, growth_rate, carrying_capacity, alpha=1, beta=0, gamma=0
):
    return general_logistic_growth(population, growth_rate, carrying_capacity, 1, 1, 1)


def richards_growth(population, growth_rate, carrying_capacity, alpha, beta, gamma):
    return general_logistic_growth(
        population, growth_rate, carrying_capacity, 1, beta, 1
    )


def blumberg_growth(population, growth_rate, carrying_capacity, alpha, beta, gamma):
    return general_logistic_growth(
        population, growth_rate, carrying_capacity, alpha, beta, 1
    )


def gompertz_growth(population, growth_rate, carrying_capacity, alpha, beta, gamma):
    return growth_rate * population * np.log(carrying_capacity / population)


class Cell:

    def __init__(self, growth_rate, name: int):
        self.growth_rate = growth_rate
        self.name = name
        self.generation = 0

    def reproduce(self, std_growth_rate=5e-4, variation_rate=0.1):
        new_growth_rate = max(
            self.growth_rate + np.random.normal(0, std_growth_rate) * np.random.binomial(1,variation_rate), 0
        )
        new_cell = Cell(new_growth_rate, self.name)
        new_cell.generation = self.generation + 1
        return new_cell


def get_proportion_per_name(cell_batch):
    names = [cell.name for cell in cell_batch]
    proportions = [names.count(name) / len(names) for name in range(MAX_NAME)]
    return proportions


def gillespie_algorithm(
    initial_cells,
    total_time,
    growth_rate_model,
    carrying_capacity,
    alpha=1,
    beta=0,
    gamma=0,
    std_growth_rate=5e-4,
    variation_rate=0.1,
    death_rate=0.5,
    get_proportions=False,
    chemostat = (-1,-1)
):
    current_population = len(initial_cells)
    populations = np.array([current_population])
    n_events, n_death, n_born = 0, 0, 0
    time = 0
    timesteps = [time]

    # Initializing current state variables
    cell_batch = initial_cells
    current_rates = np.array([cell.growth_rate for cell in cell_batch])
    sum_rates = np.sum(current_rates)
    mean_rate = sum_rates / current_population

    rate_evolution = [mean_rate]

    total_growth_rate = growth_rate_model(
        current_population, mean_rate, carrying_capacity, alpha, beta, gamma
    )
    theorical_growth_rate_evolution = [
        total_growth_rate / current_population - death_rate
    ]

    total_death_rate = death_rate * current_population

    generations = [0] * current_population
    sum_generation = 0
    mean_generation = 0
    mean_generation_evolution = [0]

    proportions_per_name = []

    chemostat_time, chemostat_rate = chemostat[0], chemostat[1]
    use_chemostat = False
    if chemostat_time > 0 and chemostat_rate > 0:
        next_chemostat_time = chemostat_time
        use_chemostat = True


    if get_proportions:
        proportions_per_name = [get_proportion_per_name(cell_batch)]

    while time < total_time:
        # Checking if the population is too large
        if current_population > 15000:
            # Printing statistics and histograms and then breaking the loop
            print("overpopulation")
            print(f"New Stop time: {time}")

            break
        
        if use_chemostat and time > next_chemostat_time:
            n_removed = int(chemostat_rate * current_population)
            if n_removed > 0:
                removed_indexes = np.random.choice(current_population, n_removed, replace=False)
                cell_batch = [cell_batch[i] for i in range(current_population) if i not in removed_indexes]
                current_population -= n_removed
                current_rates = np.array([cell.growth_rate for cell in cell_batch])
                sum_rates = np.sum(current_rates)
                mean_rate = sum_rates / current_population
                total_growth_rate = growth_rate_model(
                    current_population, mean_rate, carrying_capacity, alpha, beta, gamma
                )
                total_death_rate = death_rate * current_population
                next_chemostat_time += chemostat_time
                sum_generation = sum([cell.generation for cell in cell_batch])
                mean_generation = sum_generation / current_population
        # Generating the next time step
        dt = np.random.exponential(1 / (total_growth_rate + total_death_rate))
        time += dt

        # Calculating probabilities for each event
        death_birth_probability = [
            total_growth_rate / (total_growth_rate + total_death_rate),
            total_death_rate / (total_growth_rate + total_death_rate),
        ]

        # Choosing the event to occur based on the probabilities
        event = np.random.choice(2, p=death_birth_probability)
        n_events += 1

        # Handling event: death
        if event == 1:
            current_population -= 1

            # Checking for extinction
            if current_population == 0:
                print("extinction")
                print(f"New Stop time: {time}")
                break
            dead_cell_index = np.random.choice(current_population)
            dead_cell = cell_batch.pop(dead_cell_index)
            populations = np.append(populations, current_population)
            current_rates = np.delete(current_rates, dead_cell_index)

            generations.pop(dead_cell_index)

            sum_rates -= dead_cell.growth_rate
            mean_rate = sum_rates / current_population
            total_growth_rate = growth_rate_model(
                current_population, mean_rate, carrying_capacity, alpha, beta, gamma
            )
            total_death_rate -= death_rate

            sum_generation -= dead_cell.generation
            mean_generation = sum_generation / current_population

            n_death += 1
        else:
            # Handling event: birth
            current_population += 1
            probabilities = current_rates / sum_rates
            new_cell_index = np.random.choice(len(current_rates), p=probabilities)
            new_cell = cell_batch[new_cell_index].reproduce(std_growth_rate, variation_rate)
            cell_batch.append(new_cell)
            n_born += 1
            populations = np.append(populations, current_population)
            total_death_rate += death_rate
            sum_rates += new_cell.growth_rate
            mean_rate = sum_rates / current_population
            current_rates = np.append(current_rates, new_cell.growth_rate)
            total_growth_rate = growth_rate_model(
                current_population, mean_rate, carrying_capacity, alpha, beta, gamma
            )
            generations.append(new_cell.generation)
            sum_generation += new_cell.generation
            mean_generation = sum_generation / current_population

        # Appending time and data for plotting
        timesteps.append(time)

        rate_evolution.append(mean_rate)
        theorical_growth_rate_evolution.append(
            total_growth_rate / current_population - death_rate
        )
        mean_generation_evolution.append(mean_generation)
        if get_proportions:
            proportions_per_name.append(get_proportion_per_name(cell_batch))

    print(f"stop at time {time}")
    print(f"N_death : {n_death}, N_born : {n_born}, N_events : {n_events}\n")

    return (
        timesteps,
        populations,
        np.array(rate_evolution),
        np.array(theorical_growth_rate_evolution),
        np.array(mean_generation_evolution),
        proportions_per_name,
        time,
    )


# Run the Gillespie algorithm for N_SIMULATIONS
def run_gillespie_simulations(
    N_simulations,
    initial_population,
    growth_rate_model,
    total_time,
    carrying_capacity,
    death_rate,
    std_growth_rate,
    variation_rate,
    get_proportion_per_name_list=False,
    chemostat = (-1,-1)
):
    initial_cells = [
        [Cell(initial_growth_rate, i % MAX_NAME) for i in range(initial_population)]
        for _ in range(N_simulations)
    ]
    stop_time = total_time
    simulation_results = []
    for i in range(N_simulations):
        print(f"Simulation {i + 1}/{N_SIMULATIONS}")
        np.random.seed(i)
        simulation_results.append(
            gillespie_algorithm(
                initial_cells[i],
                stop_time,
                growth_rate_model,
                carrying_capacity,
                death_rate=death_rate,
                std_growth_rate=std_growth_rate,
                variation_rate=variation_rate,
                get_proportions=get_proportion_per_name_list,
                chemostat = chemostat
            )
        )
        stop_time = min(stop_time, simulation_results[-1][-1])
    return simulation_results, stop_time


def run_multiple_gillespie_simulations(
    N_simulations,
    initial_population,
    growth_rate_model,
    total_time,
    carrying_capacity,
    death_rate,
    std_growth_rate_list,
    variation_rate,
    get_proportion_per_name_list=False,
    chemostat = (-1,-1)
):
    all_simulation_results = []
    stop_time = total_time
    initial_cells = [
        [Cell(initial_growth_rate, i % MAX_NAME) for i in range(initial_population)]
        for _ in range(N_simulations)
    ]
    for std_growth_rate in std_growth_rate_list:
        simulation_results, stop_time = run_gillespie_simulations(
            N_simulations,
            initial_population,
            growth_rate_model,
            stop_time,
            carrying_capacity,
            death_rate,
            std_growth_rate,
            variation_rate,
            get_proportion_per_name_list,
            chemostat = chemostat
        )
        all_simulation_results.append((simulation_results, std_growth_rate))
    return all_simulation_results, stop_time


"""""

cProfile.run('run_gillespie_simulations(N_SIMULATIONS,    initial_population,    exponential_growth,    total_time,    MAX_POPULATION,    death_rate,    std_growth_rate,)', sort='tottime')

""" ""

# Execute the Gillespie algorithm
start = time.time()
all_simulation_results, stop_time = run_multiple_gillespie_simulations(
    N_SIMULATIONS,
    initial_population,
    exponential_growth,
    total_time,
    MAX_POPULATION,
    death_rate,
    std_growth_rate_list,
    variation_rate,
    get_proportion_per_name_list,
    chemostat = chemostat
)
stop = time.time()

print(f"Execution time gillespies: {stop - start}")


# Postprocessing the data for plotting
start = time.time()
all_timesteps = [
    [result[0] for result in simulation_results[0]]
    for simulation_results in all_simulation_results
]

# Merge and sort all timesteps
merged_timesteps = np.concatenate(
    [np.concatenate(all_timesteps[k]) for k in range(len(std_growth_rate_list))]
)
sorted_merged_timesteps = np.sort(merged_timesteps)

# Delete duplicates
merged_timesteps = np.unique(sorted_merged_timesteps)

# Keep only the timesteps that are present in all simulations, so before the last timestep
timesteps = merged_timesteps[merged_timesteps < stop_time]

populations_list = [
    [result[1] for result in simulation_results[0]]
    for simulation_results in all_simulation_results
]
rate_evolution_list = [
    [result[2] for result in simulation_results[0]]
    for simulation_results in all_simulation_results
]
theorical_growth_rate_evolution_list = [
    [result[3] for result in simulation_results[0]]
    for simulation_results in all_simulation_results
]
mean_generation_list = [
    [result[4] for result in simulation_results[0]]
    for simulation_results in all_simulation_results
]

# proportion_per_name_list = [result[5] for result in simulation_results]
all_simulation_results = None
# Array to store the results for each simulation
extended_results = np.empty(
    (len(timesteps), len(std_growth_rate_list), N_SIMULATIONS, 4)
)

# For each simulation, find the indices of the timesteps in the merged timesteps
for k in range(len(std_growth_rate_list)):
    for i in range(N_SIMULATIONS):
        indices = np.searchsorted(all_timesteps[k].pop(), timesteps)
        indices[indices == 0] = 1  # Replace index 0 with 1 to avoid negative indices
        indices = indices - 1  # Decrement by 1 to get the correct index
        # Fill the extended results array with the values of the current simulation
        extended_results[:, k, i, 0] = populations_list[k].pop()[indices]
        extended_results[:, k, i, 1] = rate_evolution_list[k].pop()[indices]
        extended_results[:, k, i, 2] = theorical_growth_rate_evolution_list[k].pop()[
            indices
        ]
        extended_results[:, k, i, 3] = mean_generation_list[k].pop()[indices]
        stop = time.time()
        print(f"Execution time data treatment : {stop - start}")


# Compute the average over all simulations
start = time.time()
average_populations = np.mean(extended_results[:, :, :, 0], axis=2)
average_rate_evolution = np.mean(extended_results[:, :, :, 1], axis=2)
average_theorical_growth_rate_evolution = np.mean(extended_results[:, :, :, 2], axis=2)
average_mean_generation = np.mean(extended_results[:, :, :, 3], axis=2)
extended_results = None
stop = time.time()
print(f"Execution time data treatment 2 : {stop - start}")

"""""   
#Compute effective growth rate ( from the population curve) takes time and is not readable without smoothing
dt_index =  0
dt_diff_indexes = []
last_index = 0
for i in range( len(timesteps)):
        while dt_index < len(timesteps) - 1 and timesteps[dt_index] - timesteps[i] < dt:
            dt_index += 1
        if dt_index >= len(timesteps) - 1 or i == len(timesteps) - 1:
            last_index = i 
            break
        dt_diff_indexes.append(dt_index)
local_timesteps = timesteps[:last_index]

stop = time.time()
print(f"Execution time data treatment 3 : {stop - start}")
effective_growth_rate = [(average_populations[dt_diff_indexes[i]] - average_populations[i])/(average_populations[i] * (timesteps[dt_diff_indexes[i]] - local_timesteps[i])) for i in range(len(dt_diff_indexes) )]
stop = time.time()
print(f"Execution time data treatment 4 : {stop - start}")

# Smooth the data : Does not works
def smooth_data(data, timesteps, smooth_coefficient : int):
    assert len(data) == len(timesteps)
    smoothed_data = [None] * (len(data) - smooth_coefficient)
    steps = [timesteps[i] - timesteps[i - 1] for i in range(1, len(data))]
    total_time = sum(steps[:smooth_coefficient])
    normalized_mean_data = sum([steps[i] * data[i] for i in range(smooth_coefficient)] )
    for i in range(smooth_coefficient, len(data)):    
            smoothed_data[i-smooth_coefficient] = normalized_mean_data / total_time
            total_time += steps[i-1] - steps[i - smooth_coefficient -1]
            normalized_mean_data = normalized_mean_data - data[i - smooth_coefficient] * steps[i - smooth_coefficient-1] + data[i] * steps[i-1]
    return smoothed_data

smooth_growth_rate = smooth_data(effective_growth_rate, local_timesteps, smooth_coefficient)
"""
stop = time.time()

print(f"Execution time data treatment 5 : {stop - start}")
print("N_timesteps: ", len(timesteps))


# Plot the curves

plt.figure()
plt.plot(timesteps, average_populations)
plt.title("Average Population Over Time")
plt.xlabel("Time")
plt.ylabel("Average Population")


plt.figure()
plt.plot(timesteps, average_rate_evolution)
plt.title("Average Rate Evolution Over Time")
plt.xlabel("Time")
plt.ylabel("Average Rate Evolution")
plt.grid(True)

plt.figure()
plt.plot(
    timesteps, average_theorical_growth_rate_evolution, label="Theorical Growth Rate"
)
# plt.plot(local_timesteps[smooth_coefficient//2: - smooth_coefficient//2], smooth_growth_rate, label="Effective Growth Rate")
plt.title("Average Effective Growth Rate Evolution Over Time")
plt.xlabel("Time")
plt.ylabel("Average Effective Growth Rate Evolution")
plt.grid(True)
plt.legend()


plt.figure()
plt.plot(timesteps, average_mean_generation)
plt.title("Average Mean Generation Over Time")
plt.xlabel("Time")
plt.ylabel("Average Mean Generation")

plt.figure()
plt.plot(average_mean_generation, average_rate_evolution)
plt.title("Average Inherent Growth Rate Evolution Over Mean Generation")
plt.xlabel("Mean Generation")
plt.ylabel("Average Effective Growth Rate Evolution")
plt.grid(True)

plt.show()

"""""


# Non viable code


import numpy as np

def broadcast_gillespie_algorithm(initial_cells,
                                   total_time,
                                   growth_rate_model,
                                   carrying_capacity,
                                   alpha=1,
                                   beta=0,
                                   gamma=0,
                                   std_growth_rate=5e-4,
                                   death_rate=0.5,
                                   N_simulations=10):

    initial_population = len(initial_cells)

    populations = [np.array([initial_population] * N_simulations)]
    rate_evolution = np.full((1, N_simulations), np.mean([cell.growth_rate for cell in initial_cells]))
    mean_generation = np.zeros((1, N_simulations))

    cell_batch = [initial_cells.copy() for _ in range(N_simulations)]
    generations = np.zeros((N_simulations, initial_population))
    current_rates = [np.array([cell.growth_rate for cell in sim]) for sim in cell_batch]
    sum_rates = np.array([np.sum(rates) for rates in current_rates])
    mean_rates = sum_rates / initial_population

    total_growth_rates = growth_rate_model(initial_population, mean_rates, carrying_capacity, alpha, beta, gamma)
    total_death_rates = np.full(N_simulations, death_rate * initial_population)
    theorical_growth_rate_evolution = [list(total_growth_rates / initial_population - death_rate)]

    current_time = 0
    timesteps = np.array([0])

    next_time_steps = np.random.exponential(1 / (total_growth_rates + total_death_rates))
    N_deaths = np.zeros(N_simulations, dtype=int)
    N_borns = np.zeros(N_simulations, dtype=int)

    while current_time < total_time:
        sim_index = np.argmin(next_time_steps)
        current_time = next_time_steps[sim_index]

        death_birth_events = np.random.rand() < (total_death_rates / (total_growth_rates + total_death_rates))

        # Handling event: death
        if np.any(death_birth_events):
            dead_cell_index = np.random.choice(len(current_rates[sim_index]))

            # Remove dead cells
            cell_batch[sim_index] = np.delete(cell_batch[sim_index], dead_cell_index)
            sum_rates[sim_index] -= current_rates[sim_index][dead_cell_index]
            current_rates[sim_index] = np.delete(current_rates[sim_index], dead_cell_index)
            mean_rates[sim_index] = sum_rates[sim_index] / len(current_rates[sim_index])
            total_growth_rates[sim_index] = growth_rate_model(len(current_rates[sim_index]), mean_rates[sim_index], carrying_capacity, alpha, beta, gamma)
            total_death_rates[sim_index] -= death_rate
            N_deaths[sim_index] += 1

            

        # Handling event: birth
        else:
            probabilities = current_rates[sim_index] / sum_rates[sim_index]
            new_cell_index = np.random.choice(len(current_rates[sim_index]), p=probabilities)

            new_cell = cell_batch[sim_index][new_cell_index].reproduce(std_growth_rate)
            cell_batch[sim_index] = np.append(cell_batch[sim_index], new_cell)
            sum_rates[sim_index] += new_cell.growth_rate
            mean_rates[sim_index] = sum_rates[sim_index] / len(cell_batch[sim_index])
            current_rates[sim_index] = np.append(current_rates[sim_index], new_cell.growth_rate)
            total_growth_rates[sim_index] = growth_rate_model(len(cell_batch[sim_index]), mean_rates[sim_index], carrying_capacity, alpha, beta, gamma)
            total_death_rates[sim_index] += death_rate
            N_borns[sim_index] += 1

            

        # Appending time and data for plotting
        current_time_steps = np.random.exponential(1 / (total_growth_rates + total_death_rates))
        timesteps = np.append(timesteps, current_time + current_time_steps)
        next_time_steps[sim_index] = current_time + current_time_steps[sim_index]

        rate_evolution = np.append(rate_evolution, [mean_rates], axis=0)
        theorical_growth_rate_evolution.append([total_growth_rates[i] / len(cell_batch[i]) - death_rate for i in range(N_simulations)])
        mean_generation = np.append(mean_generation, [np.mean(generations, axis=1)], axis=0)
        populations.append(np.array([len(cell_batch[i]) for i in range(N_simulations)]))

    print("Simulation completed.")
    print(f"Deaths: {N_deaths}, Births: {N_borns}")

    return timesteps, populations, rate_evolution, theorical_growth_rate_evolution, mean_generation


start = time.time()
(timesteps, populations, rate_evolution, theorical_growth_rate_evolution, mean_generation) = broadcast_gillespie_algorithm(initial_cells, total_time, exponential_growth, MAX_POPULATION, death_rate=death_rate, N_simulations=N_SIMULATIONS, std_growth_rate=std_growth_rate)
stop = time.time()
print(f"Execution time: {stop - start}")






# Extraire les valeurs pour le tracé
start = time.time()
populations = np.array(populations)
average_populations = np.mean(populations, axis=0)
average_rate_evolution = np.mean(rate_evolution, axis=1)
average_theorical_growth_rate_evolution =   np.mean(theorical_growth_rate_evolution, axis=1)
average_mean_generation = np.mean(mean_generation, axis=1)
stop = time.time()

print(f"Execution time data treatment: {stop - start}")
print(f"N_timesteps: {len(timesteps)}")
# Tracer les courbes

plt.figure()
plt.plot(timesteps, average_populations)
plt.title('Average Population Over Time')
plt.xlabel('Time')
plt.ylabel('Average Population')

plt.figure()
plt.plot(timesteps, average_rate_evolution)
plt.title('Average Rate Evolution Over Time')
plt.xlabel('Time')
plt.ylabel('Average Rate Evolution')

plt.figure()
plt.plot(timesteps, average_theorical_growth_rate_evolution)
plt.title('Average Effective Growth Rate Evolution Over Time')
plt.xlabel('Time')
plt.ylabel('Average Effective Growth Rate Evolution')
plt.grid(True)

plt.figure()
plt.plot(timesteps, average_mean_generation)
plt.title('Average Mean Generation Over Time')
plt.xlabel('Time')
plt.ylabel('Average Mean Generation')

plt.show()
"""
