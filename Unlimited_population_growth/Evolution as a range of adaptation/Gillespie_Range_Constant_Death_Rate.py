from typing import Optional
import numpy as np
import time
import matplotlib.pyplot as plt


"""
Gillespie algorithm for the evolution of a population of cells
with a constant death rate and a constant growth rate.
An evolution is a range of values for which the cell is adapted.
"""


# Global Constants
DISTANT_COEFF = 4.8
RANGE_COEFF = 2
MAX_GROWTH_RATE = 1



# Function to calculate the distance between a value and an interval
def distant_range(min_val, max_val, e):
    return max(0, max(min_val - e, e - max_val))


# Function to plot the histogram of the distribution of genotypes
def print_hist(genotypes, time):
    # Initialization of parameters for the histogram
    num_point = 1000
    min_val = np.min([genotype[0] for genotype in genotypes])
    max_val = np.max([genotype[1] for genotype in genotypes])
    value = np.zeros(num_point)
    points = np.linspace(min_val, max_val, num_point)

    # Calculation of the histogram
    for gen in genotypes:
        min_g, max_g = gen
        min_point = int((min_g - min_val) / (max_val - min_val) * num_point)
        max_point = int((max_g - min_val) / (max_val - min_val) * num_point)
        value[min_point:max_point] += 1
    value = [v / len(genotypes) for v in value]

    # Calculation of mean values
    mean_min_values = np.mean([genotype[0] for genotype in genotypes])
    mean_max_values = np.mean([genotype[1] for genotype in genotypes])
    mean_value = (mean_max_values + mean_min_values) / 2

    # Plotting the histogram
    plt.plot(points, value)
    plt.axvline(x=points[np.argmax(value)], color="r", label="Maximal density")
    plt.axvline(x=mean_min_values, color="g", label="Mean of min value")
    plt.axvline(x=mean_max_values, color="b", label="Mean of max value")
    plt.axvline(x=mean_value, color="y", label="Mean value")
    plt.legend()
    plt.xlabel("Values")
    plt.ylabel("Probability Density")
    plt.title(f"Distribution of values / population: {len(genotypes)} / time : {time}")
    plt.grid(True)
    plt.show()
    return mean_value


# Function to calculate growth rate based on genotype, phenotype, and condition
def growth_rate(genotype, phenotype, condition):
    min_g, max_g = genotype
    mini, maxi = min_g, max_g
    if phenotype is not None:
        min_p, max_p = phenotype
        mini, maxi = min(min_p, min_g), max(max_p, max_g)
    return (
        MAX_GROWTH_RATE
        / ((1 + np.abs(max_g - min_g) * RANGE_COEFF))
        * np.exp(-DISTANT_COEFF * (distant_range(mini, maxi, condition)))
    )


# Class representing evolutive cells
class EvolutiveCells:
    def __init__(self, first_evolution: tuple[float, float], conditions=0):
        self.phenotype = None  # phenotype is None if the cell has not adapted to the condition, change often
        self.genotype = first_evolution  # genotype is the range for which the cell is adapted, change rarely
        self.growth_rate = growth_rate(first_evolution, None, conditions)
        self.absolute_generation = 0  # number of generations since the beginning
        self.generation_on_same_evolution = (
            0  # number of generations since the genotype has changed
        )
        self.evolution_generation = 0  # number of evolutions since the beginning

    def copy(self, conditions):
        new_cell = EvolutiveCells(self.genotype, conditions)
        new_cell.phenotype = self.phenotype
        new_cell.absolute_generation = self.absolute_generation + 1
        new_cell.evolution_generation = self.evolution_generation
        new_cell.generation_on_same_evolution = self.generation_on_same_evolution + 1
        return new_cell


# Function implementing the Gillespie algorithm for evolution simulation
def gillespie_algorithm(
    initial_evolutions,
    total_time,
    condition_profile,
    n_plot=5,
    evolutions_probability=0.1,
    adaptation_probability=0.1,
    evol_probability_without_adaptation=0.1,
    probability_to_loose_adaptation=1 / 5,
    death_rate=0.5,
):
    np.random.seed(42)
    populations = np.array([len(initial_evolutions)])
    n_evol, n_adapt, n_evol_wa = 0, 0, 0
    n_events, n_death, n_born = 0, 0, 0
    time = 0
    timesteps = [time]

    np.random.seed(0)
    # Extracting the first condition and its time from the condition profile
    change_time, condition = condition_profile.pop(0)

    # Calculating the time interval between each plot
    time_between_plot = total_time / n_plot
    time_next_plot = time_between_plot

    # Initializing current state variables
    current_evolutions = list(initial_evolutions)
    mean_value = print_hist(current_evolutions, 0)
    cell_batch = [
        EvolutiveCells(evolution, condition) for evolution in initial_evolutions
    ]
    current_rates = np.array([cell.growth_rate for cell in cell_batch])
    current_rates = np.insert(current_rates, 0, death_rate * len(cell_batch))
    rate_evolution = [np.mean(current_rates[1:])]
    total_rate = np.sum(current_rates)

    mean_values = [mean_value]
    mean_range = [
        np.mean([genotype[1] - genotype[0] for genotype in current_evolutions])
    ]

    absolute_generations = [cell.absolute_generation for cell in cell_batch]
    evolution_gererations = [cell.evolution_generation for cell in cell_batch]
    generations_on_same_evolution = [
        cell.generation_on_same_evolution for cell in cell_batch
    ]

    mean_generation = [np.mean(absolute_generations)]
    mean_evolution_generation = [np.mean(evolution_gererations)]
    mean_generation_on_same_evolution = [np.mean(generations_on_same_evolution)]

    genetic_diversity = [len(set(current_evolutions)) / len(current_evolutions)]

    while time < total_time:
        # Updating condition if time surpasses the change time and there are more conditions left
        if time > change_time and condition_profile:
            change_time, condition = condition_profile.pop(0)
            current_rates = np.array(
                [
                    growth_rate(cell.genotype, cell.phenotype, condition)
                    for cell in cell_batch
                ]
            )
            current_rates = np.insert(current_rates, 0, death_rate * len(cell_batch))
            total_rate = np.sum(current_rates)

        # Checking for extinction
        if len(cell_batch) == 0:
            print("extinction")
            break

        # Checking if the population is too large
        if len(cell_batch) > 13000:
            # Printing statistics and histograms and then breaking the loop
            print(
                f"N_evolution : {n_evol}, N_adaptation : {n_adapt}, N_death : {n_death}, N_born : {n_born}, N_events : {n_events}, N_evol_without_adaptation : {n_evol_wa}\n"
            )

            print(
                f"Mean generation : {np.mean(absolute_generations)},"
                + f"Max generation : {max(absolute_generations)},"
                + f"Min generation : {min(absolute_generations)},"
                + f"Median generation : {np.median(absolute_generations)} ,"
                + f"Std generation : {np.std(absolute_generations)} \n"
            )

            print(
                f"Mean evolution generation : {np.mean(evolution_gererations)},"
                + f"Max evolution generation : {max(evolution_gererations)},"
                + f"Min evolution generation : {min(evolution_gererations)},"
                + f"Median evolution generation : {np.median(evolution_gererations)} ,"
                + f"Std evolution generation : {np.std(evolution_gererations)} \n"
            )

            print(
                f"Mean generation on same evolution : {np.mean(generations_on_same_evolution)},"
                + f"Max generation on same evolution : {max(generations_on_same_evolution)},"
                + f"Median generation on same evolution : {np.median(generations_on_same_evolution)} ,"
                + f"Std generation on same evolution : {np.std(generations_on_same_evolution)} \n"
            )

            _ = print_hist(current_evolutions, time)
            break

        # Generating the next time step
        dt = np.random.exponential(1 / total_rate)
        time += dt

        # Calculating probabilities for each event
        probabilities = current_rates / total_rate

        # Choosing the event to occur based on the probabilities
        event = np.random.choice(len(current_rates), p=probabilities)
        n_events += 1

        # Tracking genetic diversity
        genetic_diversity.append(len(set(current_evolutions)) / len(current_evolutions))

        # Handling event: death
        if event == 0:
            dead_cell_index = np.random.choice(len(cell_batch))
            dead_cell = cell_batch.pop(dead_cell_index)

            # Checking for extinction
            if len(cell_batch) == 0:
                print("extinction")
                break

            populations = np.append(populations, len(cell_batch))

            total_rate -= death_rate + current_rates[dead_cell_index + 1]
            mean_value = (
                mean_value * (populations[-1] + 1)
                - (dead_cell.genotype[1] + dead_cell.genotype[0]) / 2
            ) / (populations[-1])

            current_rates = np.delete(current_rates, dead_cell_index + 1)
            current_rates[0] -= death_rate
            current_evolutions.pop(dead_cell_index)
            absolute_generations.pop(dead_cell_index)
            evolution_gererations.pop(dead_cell_index)
            generations_on_same_evolution.pop(dead_cell_index)
            n_death += 1
        else:
            # Handling event: birth
            new_cell = cell_batch[event - 1].copy(condition)
            cell_batch.append(new_cell)
            n_born += 1
            populations = np.append(populations, len(cell_batch))
            current_rates[0] += death_rate
            total_rate += death_rate
            change_probabilities = np.random.uniform(0, 1, 4)

            # Handling evolution event without adaptation
            if change_probabilities[0] < evol_probability_without_adaptation:
                new_cell.genotype = np.random.normal(
                    new_cell.genotype[0], 0.05
                ), np.random.normal(new_cell.genotype[1], 0.05)
                new_cell.evolution_generation += 1
                new_cell.generation_on_same_evolution = 0
                n_evol += 1
                n_evol_wa += 1

            # Handling adaptation event
            if change_probabilities[1] < adaptation_probability:
                new_cell.phenotype = np.random.normal(
                    new_cell.genotype[0], 0.1
                ), np.random.normal(new_cell.genotype[1], 0.1)
                n_adapt += 1

            # Handling probability to lose adaptation event
            if change_probabilities[2] < probability_to_loose_adaptation:
                new_cell.phenotype = None

            # Handling evolution event with adaptation
            if (
                change_probabilities[3] < evolutions_probability
                and new_cell.phenotype is not None
            ):
                mini, maxi = np.random.normal(
                    new_cell.phenotype[0], 0.05
                ), np.random.normal(new_cell.phenotype[1], 0.05)
                if mini < maxi:
                    new_cell.genotype = mini, maxi
                else:
                    new_cell.genotype = maxi, mini
                new_cell.evolution_generation += 1
                new_cell.generation_on_same_evolution = 0
                n_evol += 1

            # Updating mean value after the event
            mean_value = (
                mean_value * (populations[-1] - 1)
                + (new_cell.genotype[1] + new_cell.genotype[0]) / 2
            ) / (populations[-1])

            # Updating data for new cell
            current_evolutions.append(new_cell.genotype)
            current_rates = np.append(
                current_rates,
                growth_rate(new_cell.genotype, new_cell.phenotype, condition),
            )
            total_rate += current_rates[-1]
            absolute_generations.append(new_cell.absolute_generation)
            evolution_gererations.append(new_cell.evolution_generation)
            generations_on_same_evolution.append(new_cell.generation_on_same_evolution)

        # Appending time and data for plotting
        timesteps.append(time)
        mean_generation.append(np.mean(absolute_generations))
        mean_evolution_generation.append(np.mean(evolution_gererations))
        mean_generation_on_same_evolution.append(np.mean(generations_on_same_evolution))
        rate_evolution.append(np.mean(current_rates[1:]))
        mean_range.append(
            np.mean([genotype[1] - genotype[0] for genotype in current_evolutions])
        )
        mean_values.append(mean_value)

        # Checking if it's time to plot
        if time > time_next_plot:
            # Printing statistics and histograms and then breaking the loop
            print(
                f"N_evolution : {n_evol}, N_adaptation : {n_adapt}, N_death : {n_death}, N_born : {n_born}, N_events : {n_events}, N_evol_without_adaptation : {n_evol_wa}\n"
            )

            print(
                f"Mean generation : {np.mean(absolute_generations)},"
                + f"Max generation : {max(absolute_generations)},"
                + f"Min generation : {min(absolute_generations)},"
                + f"Median generation : {np.median(absolute_generations)} ,"
                + f"Std generation : {np.std(absolute_generations)} \n"
            )

            print(
                f"Mean evolution generation : {np.mean(evolution_gererations)},"
                + f"Max evolution generation : {max(evolution_gererations)},"
                + f"Min evolution generation : {min(evolution_gererations)},"
                + f"Median evolution generation : {np.median(evolution_gererations)} ,"
                + f"Std evolution generation : {np.std(evolution_gererations)} \n"
            )

            print(
                f"Mean generation on same evolution : {np.mean(generations_on_same_evolution)},"
                + f"Max generation on same evolution : {max(generations_on_same_evolution)},"
                + f"Median generation on same evolution : {np.median(generations_on_same_evolution)} ,"
                + f"Std generation on same evolution : {np.std(generations_on_same_evolution)} \n"
            )

            _ = print_hist(current_evolutions, time)
            print(f"Time : {time}")
            time_next_plot += time_between_plot

    return (
        timesteps,
        populations,
        current_evolutions,
        rate_evolution,
        mean_range,
        mean_generation,
        mean_evolution_generation,
        mean_generation_on_same_evolution,
        genetic_diversity,
        mean_values,
    )


"""
period_duration = 50 
condition_profile = [(t/10, np.sin(2 * np.pi * t/(10*period_duration*2))**2) for t in range(1500)]
condition_profile.extend([(150+ t/10, 0.3 +np.sin(2 * np.pi * t/(10*period_duration*2))**2) for t in range(1500)])
condition_profile.extend([(300+ t/10, np.sin(2 * np.pi * t/(10*period_duration*2))**2) for t in range(1500)])
condition_profile.extend([(450+ t/10, 0.3 +np.sin(2 * np.pi * t/(10*period_duration*2))**2) for t in range(1500)])
"""
period_duration = 1
condition_profile = [(40, 0.8)]
condition_profile.append((80, 0.28))
condition_profile.append((120, 0.85))

# Parameters
population = 925
adaptation_probability = 0.1
evolution_probability = 0.1
evolution_without_adaptation_probability = 5e-4
mean_time_to_loose_adaptation = 4
probability_to_loose_adaptation = 1 / mean_time_to_loose_adaptation
maxs = np.random.uniform(0.5, 1.2, population)
mins = np.random.uniform(-0.2, 0.5, population)
initial_evolutions = [(mins[i], maxs[i]) for i in range(population)]
death_rate = 0.47
total_time = 250


start = time.time()
(
    timesteps,
    populations,
    current_evolutions,
    rate_evolution,
    mean_range,
    mean_generation,
    mean_evolution_generation,
    mean_generation_on_same_evolution,
    genetic_diversity,
    mean_values,
) = gillespie_algorithm(
    initial_evolutions,
    total_time,
    condition_profile,
    evolutions_probability=evolution_probability,
    adaptation_probability=adaptation_probability,
    evol_probability_without_adaptation=evolution_without_adaptation_probability,
    probability_to_loose_adaptation=probability_to_loose_adaptation,
    death_rate=death_rate,
    n_plot=1,
)
stop = time.time()


print(f"Execution time (gilespie algorithm): {stop - start}s")
print(f"timesteps : {len(timesteps)}")

start = time.time()
normalized_growth_rates = [
    rate_evolution[i] * (timesteps[i + 1] - timesteps[i])
    for i in range(len(rate_evolution) - 1)
]
averages = np.array([])
end_indexes = []
time_periods = []
end_index = 0

# Useful if the condition profile is periodic

for start_index in range(len(timesteps)):
    while not (timesteps[end_index] - timesteps[start_index] >= period_duration):
        end_index += 1
        if end_index == len(timesteps) - 1:
            break
    if end_index == len(timesteps) - 1:
        break
    end_indexes.append(end_index)
    time_periods.append(timesteps[end_index])
stop = time.time()
print(f"Execution time (periods indices calculation): {stop - start}s")
start = time.time()

normalized_growth_rates = np.array(normalized_growth_rates)
end_indices = np.array(end_indexes)
cumulative_growth_sum = np.cumsum(normalized_growth_rates)
stop = time.time()
print(f"Execution time (cumulative sum calculation): {stop - start}s")
start = time.time()
averages = (
    cumulative_growth_sum[end_indexes] - cumulative_growth_sum[range(len(end_indexes))]
)

# Compute the average growth rate on one period
averages = averages / (period_duration)
stop = time.time()

# Compute the genetic diversity after 10s, because the genetic diversity is not interesting at the beginning : it is always 1
indices_10s = [i for i in range(len(timesteps)) if timesteps[i] >= 10]
genetic_diversity_10s = [genetic_diversity[i] for i in indices_10s]
time_10s = [timesteps[i] for i in indices_10s]
print(f"Execution time (average calculation): {stop - start}s")
# Plot the results

plt.figure()
plt.plot(timesteps, populations)
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Population evolution")


plt.figure()
plt.plot(timesteps, rate_evolution)
plt.plot(timesteps, [death_rate] * len(timesteps), label="Death rate")
plt.xlabel("Time")
plt.ylabel("Growth rate")
plt.title("Growth Rate evolution")

plt.figure()
plt.plot(timesteps, mean_range)
plt.xlabel("Time")
plt.ylabel("Mean range")
plt.title("Mean range evolution")

plt.figure()
plt.plot(time_periods, averages)
plt.xlabel("Time")
plt.ylabel("Average growth rate on one period")

plt.figure()
plt.plot(timesteps, mean_generation)
plt.xlabel("Time")
plt.ylabel("Mean generation")
plt.title("Mean cell generation")

plt.figure()
plt.plot(timesteps, mean_evolution_generation)
plt.xlabel("Time")
plt.ylabel("Mean evolution generation")
plt.title("Mean evolution generation")

plt.figure()
plt.plot(timesteps, mean_generation_on_same_evolution)
plt.xlabel("Time")
plt.ylabel("Mean generation on same evolution")
plt.title("Mean generation on same evolution")

plt.figure()
plt.plot(time_10s, genetic_diversity_10s)
plt.xlabel("Time")
plt.ylabel("Genetic diversity")
plt.title("Genetic diversity")

plt.figure()
plt.plot(timesteps, mean_values)
plt.xlabel("Time")
plt.ylabel("Mean value")
plt.title("Mean value")


plt.show()
