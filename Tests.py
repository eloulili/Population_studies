import numpy as np
import matplotlib.pyplot as plt
import time
import cProfile

MAX_POPULATION = 1000
N_SIMULATIONS = 1
MAX_NAME = 1


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

    def reproduce(self, std_growth_rate=5e-4):
        new_growth_rate = max(
            self.growth_rate + np.random.normal(0, std_growth_rate), 0
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
    death_rate=0.5,
    get_proportions=False,
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
    effective_growth_rate_evolution = [
        total_growth_rate / current_population - death_rate
    ]

    total_death_rate = death_rate * current_population

    generations = [0] * current_population
    sum_generation = 0
    mean_generation = 0
    mean_generation_evolution = [0]

    proportions_per_name = []

    if get_proportions:
        proportions_per_name = [get_proportion_per_name(cell_batch)]

    while time < total_time:

        # Checking if the population is too large
        if current_population > 10000:
            # Printing statistics and histograms and then breaking the loop
            print("overpopulation")
            print(f"New Stop time: {time}")

            break

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
            new_cell = cell_batch[new_cell_index].reproduce(std_growth_rate)
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
        effective_growth_rate_evolution.append(
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
        np.array(effective_growth_rate_evolution),
        np.array(mean_generation_evolution),
        proportions_per_name,
        time,
    )


# Initialize parameters

np.random.seed(0)
initial_population = 600
total_time = 400
death_rate = 1.005
initial_growth_rate = 1.0
std_growth_rate = 0.001
initial_cells = [
    Cell(initial_growth_rate + i % MAX_NAME / 200, i % MAX_NAME)
    for i in range(initial_population)
]
get_proportion_per_name_list = False


# Run the Gillespie algorithm for N_SIMULATIONS
def run_gillespie_simulations(
    N_simulations,
    initial_population,
    growth_rate_model,
    total_time,
    carrying_capacity,
    death_rate,
    std_growth_rate,
    get_proportion_per_name_list=False,
):
    initial_cells = [
        [Cell(initial_growth_rate, i % MAX_NAME) for i in range(initial_population)]
        for _ in range(N_simulations)
    ]
    stop_time = total_time
    simulation_results = []
    for i in range(N_simulations):
        print(f"Simulation {i + 1}/{N_SIMULATIONS}")
        np.random.seed(0)
        simulation_results.append(
            gillespie_algorithm(
                initial_cells[i],
                stop_time,
                growth_rate_model,
                carrying_capacity,
                death_rate=death_rate,
                std_growth_rate=std_growth_rate,
                get_proportions=get_proportion_per_name_list,
            )
        )
        stop_time = min(stop_time, simulation_results[-1][-1])
    return simulation_results, stop_time


cProfile.run(
    "run_gillespie_simulations(N_SIMULATIONS,    initial_population,    exponential_growth,    total_time,    MAX_POPULATION,    death_rate,    std_growth_rate,)",
    sort="tottime",
)
