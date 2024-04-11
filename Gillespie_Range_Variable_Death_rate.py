from typing import Optional
import numpy as np
import time
import matplotlib.pyplot as plt
import cProfile
import pstats


DISTANT_COEFF = 30
RANGE_COEFF = 3.5
N0 = 100000
BASE_DEATH_RATE = 0.4
np.random.seed(42)


def distant_range(min, max, e):
    if max < min:
        max, min = min, max
    if e < min:
        return min - e
    elif e > max:
        return e - max
    else:
        return 0


def print_hist(genotypes):
    num_point = 1000
    min = np.min([genotype[0] for genotype in genotypes])
    max = np.max([genotype[1] for genotype in genotypes])
    value = [0 for _ in range(num_point)]
    points = np.linspace(min, max, num_point)
    for gen in genotypes:
        min_g, max_g = gen
        min_point = int((min_g - min) / (max - min) * num_point)
        max_point = int((max_g - min) / (max - min) * num_point)
        for i in range(min_point, max_point):
            value[i] += 1
    value = [v / len(genotypes) for v in value]

    plt.plot(points, value)
    plt.xlabel("Value")
    plt.ylabel("Probability density")
    plt.title("Distribution /" + " population: " + str(len(genotypes)))
    plt.grid(True)
    plt.show()


def growth_rate(genotype, phenotype, condition):
    min_g, max_g = genotype
    mini, maxi = min_g, max_g
    if phenotype is not None:
        min_p, max_p = phenotype
        mini, maxi = min(min_p, min_g), max(max_p, max_g)
    return (
        1
        / ((1 + np.abs(max_g - min_g) * RANGE_COEFF))
        * np.exp(-DISTANT_COEFF * (distant_range(mini, maxi, condition)))
    )  # Penality for the distance and the range


def death_rate(N_cells):
    return (max(0, (N_cells / N0)**2 - 1) + BASE_DEATH_RATE) * N_cells


class EvolutiveCells1D:
    def __init__(self, first_evolution: tuple[float, float], conditions=0):

        self.phenotype = (
            None  # the first element is the adaptation, the second is the duration
        )
        self.genotype = first_evolution
        self.growth_rate = growth_rate(first_evolution, None, conditions)
        self.absolute_generation = 0
        self.generation_on_same_evolution = 0
        self.evolution_generation = 0

    def copy(self, conditions):
        new_cell = EvolutiveCells1D(self.genotype, conditions)
        new_cell.phenotype = self.phenotype
        new_cell.genotype = self.genotype
        new_cell.absolute_generation = self.absolute_generation + 1
        new_cell.evolution_generation = self.evolution_generation
        new_cell.generation_on_same_evolution = self.generation_on_same_evolution + 1
        return new_cell


def gillespie_algorithm(
    initial_evolutions,
    total_time,
    condition_profile,
    n_plot=5,
    evolutions_probability=0.1,
    adaptation_probability=0.1,
    evol_probability_without_adaptation=0.1,
    probability_to_loose_adaptation=1 / 5,
):
    populations = np.array([len(initial_evolutions)])
    current_population = len(initial_evolutions)
    n_evol, n_adapt, n_evol_wa = 0, 0, 0
    n_events, n_death, n_born = 0, 0, 0
    time = 0
    timesteps = [time]

    condition_index = 0
    max_condition_index = len(condition_profile)
    change_time, condition = condition_profile[0]

    time_between_plot = total_time / n_plot
    time_next_plot = time_between_plot
    current_evolutions = list(initial_evolutions)
    print_hist(current_evolutions)
    cell_batch = [
        EvolutiveCells1D(evolution, condition) for evolution in initial_evolutions
    ]
    current_rates = np.array([cell.growth_rate for cell in cell_batch])

    initial_death_rate = death_rate(current_population)
    current_rates = np.insert(current_rates, 0, initial_death_rate)

    total_rate = np.sum(current_rates)
    sum_growth_rate = np.sum(current_rates[1:])
    rate_evolution = [sum_growth_rate / current_population]
    death_rate_evolution = [initial_death_rate/current_population]
    mean_range = [
        np.mean([genotype[1] - genotype[0] for genotype in current_evolutions])
    ]

    absolute_generations = [cell.absolute_generation for cell in cell_batch]
    sum_absolute_generations = np.sum(absolute_generations)

    evolution_gererations = [cell.evolution_generation for cell in cell_batch]
    sum_evolution_generations = np.sum(evolution_gererations)
    sum_mean_genotype = np.sum([genotype[1] + genotype[0] for genotype in current_evolutions])
    mean_genotype_center = [sum_mean_genotype / (2 * current_population)]

    generations_on_same_evolution = [
        cell.generation_on_same_evolution for cell in cell_batch
    ]
    sum_generations_on_same_evolution = np.sum(generations_on_same_evolution)

    mean_generation = [sum_absolute_generations / current_population]
    mean_evolution_generation = [sum_evolution_generations / current_population]
    mean_generation_on_same_evolution = [sum_generations_on_same_evolution / current_population]

    genetic_richness = [len(set(current_evolutions)) / len(current_evolutions)]
    while time < total_time:
        if time > change_time and condition_index < max_condition_index -1:
            condition_index += 1
            change_time, condition = condition_profile[condition_index]
            former_death_rate = current_rates[0]
            current_rates = np.array(
                [
                    growth_rate(cell.genotype, cell.phenotype, condition)
                    for cell in cell_batch
                ]
            )
            current_rates = np.insert(current_rates, 0, former_death_rate)
            total_rate = np.sum(current_rates)
            sum_growth_rate = np.sum(current_rates[1:])

        if len(cell_batch) == 0:
            print("extinction")
            break

        if len(cell_batch) > 10000:
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

            print_hist(current_evolutions)
            break
        dt = np.random.exponential(1 / total_rate)
        time += dt

        probabilities = current_rates / total_rate
        event = np.random.choice(len(current_rates), p=probabilities)
        n_events += 1

        if event == 0:  # death
            dead_cell = np.random.choice(len(cell_batch))
            cell_batch.pop(dead_cell)
            current_population -= 1

            if current_population == 0:
                print("extinction")
                break

            total_rate -= current_rates[dead_cell + 1]
            sum_growth_rate -= current_rates[dead_cell + 1]
            total_rate -= current_rates[0]

            current_rates = np.delete(current_rates, dead_cell + 1)

            current_rates[0] = death_rate(current_population)
            total_rate += current_rates[0]

            deleted_evolution = current_evolutions.pop(dead_cell)
            sum_mean_genotype -= (deleted_evolution[0] + deleted_evolution[1])

            sum_absolute_generations-=absolute_generations.pop(dead_cell)
            sum_evolution_generations-= evolution_gererations.pop(dead_cell)
            sum_generations_on_same_evolution-= generations_on_same_evolution.pop(dead_cell)

            n_death += 1
            populations = np.append(populations, current_population)
        else:
            new_cell = cell_batch[event - 1].copy(condition)
            cell_batch.append(new_cell)
            n_born += 1
            current_population += 1
            populations = np.append(populations, current_population)

            total_rate -= current_rates[0]
            current_rates[0] = death_rate(current_population)
            total_rate += current_rates[0]

            change_probabilities = np.random.uniform(0, 1, 4)

            if change_probabilities[0] < evol_probability_without_adaptation:
                new_cell.genotype = np.random.normal(
                    new_cell.genotype[0], 0.05
                ), np.random.normal(new_cell.genotype[1], 0.05)
                new_cell.evolution_generation += 1
                new_cell.generation_on_same_evolution = 0
                n_evol += 1
                n_evol_wa += 1

            if change_probabilities[1] < adaptation_probability:
                new_cell.phenotype = np.random.normal(
                    new_cell.genotype[0], 0.1
                ), np.random.normal(new_cell.genotype[1], 0.1)
                n_adapt += 1

            if change_probabilities[2] < probability_to_loose_adaptation:
                new_cell.phenotype = None

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

            current_evolutions.append(new_cell.genotype)
            current_rates = np.append(
                current_rates,
                growth_rate(new_cell.genotype, new_cell.phenotype, condition),
            )
            total_rate += current_rates[-1]
            sum_growth_rate += current_rates[-1]

            absolute_generations.append(new_cell.absolute_generation)
            evolution_gererations.append(new_cell.evolution_generation)
            generations_on_same_evolution.append(new_cell.generation_on_same_evolution)
            sum_absolute_generations+=new_cell.absolute_generation
            sum_evolution_generations+= new_cell.evolution_generation
            sum_generations_on_same_evolution+= new_cell.generation_on_same_evolution

            sum_mean_genotype += (new_cell.genotype[0] + new_cell.genotype[1])

        mean_generation.append(sum_absolute_generations / current_population)
        mean_evolution_generation.append(sum_evolution_generations / current_population)
        mean_generation_on_same_evolution.append(sum_generations_on_same_evolution / current_population)

        rate_evolution.append(sum_growth_rate / current_population)
        death_rate_evolution.append(current_rates[0]/current_population)

        mean_range.append(
            np.mean([genotype[1] - genotype[0] for genotype in current_evolutions])
        )
        mean_genotype_center.append(sum_mean_genotype/ (2 * current_population))
                                   


        timesteps.append(time)
        genetic_richness.append(len(set(current_evolutions)) / len(current_evolutions))

        if time > time_next_plot:

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

            print_hist(current_evolutions)
            print(f"Time : {time}")
            time_next_plot += time_between_plot

    return (
        timesteps,
        populations,
        current_evolutions,
        rate_evolution,
        death_rate_evolution,
        mean_range,
        mean_generation,
        mean_evolution_generation,
        mean_generation_on_same_evolution,
        genetic_richness,
        mean_genotype_center,

    )

period_duration = 50


condition_profile = []
last_condition = 0.50
for t in range(800):
    condition_profile.append((0.25 * t, last_condition))
    new_condition_change = np.random.normal(-0.0023, 0.002)
    last_condition = last_condition + new_condition_change

population = 5000
adaptation_probability = 0.1
evolution_probability = 0.1
evolution_without_adaptation_probability = 5e-4
mean_time_to_loose_adaptation = 3.5
probability_to_loose_adaptation = 1 / mean_time_to_loose_adaptation
maxs = np.random.uniform(0.5, 1.2, population)
mins = np.random.uniform(-0.2, 0.4, population)
initial_evolutions = [(mins[i], maxs[i]) for i in range(population)]
total_time = 200



start = time.time()
(
    timesteps,
    populations,
    current_evolutions,
    rate_evolution,
    death_rate_evolution,   
    mean_range,
    mean_generation,
    mean_evolution_generation,
    mean_generation_on_same_evolution,
    genetic_richness,
    mean_genotype_center,
) = gillespie_algorithm(
    initial_evolutions,
    total_time,
    condition_profile,
    evolutions_probability=evolution_probability,
    adaptation_probability=adaptation_probability,
    evol_probability_without_adaptation=evolution_without_adaptation_probability,
    probability_to_loose_adaptation=probability_to_loose_adaptation,
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
# Calculez la moyenne du taux de croissance pour chaque période de temps en divisant les sommes cumulatives par la durée de la période
averages = averages / (period_duration)
stop = time.time()

indices_10s = [i for i in range(len(timesteps)) if timesteps[i] >= 10]
genetic_richness_10s = [genetic_richness[i] for i in indices_10s]
time_10s = [timesteps[i] for i in indices_10s]


Conditions = []
index = 0
(end_time, first_condition) = condition_profile[0]

for timestep in timesteps:
    if timestep > end_time and index < len(condition_profile) - 1:
        index += 1
        (end_time, first_condition) = condition_profile[index]
    Conditions.append(first_condition)

        

print(f"Execution time (average calculation): {stop - start}s")
# Plot the results

plt.figure()
plt.plot(timesteps, populations)
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Population evolution")


plt.figure()
plt.plot(timesteps, rate_evolution, label="Growth rate")
plt.plot(timesteps, death_rate_evolution, label="Death rate")
plt.xlabel("Time")
plt.ylabel("Rate ")
plt.title("Rate evolution")
plt.legend()

plt.figure()
plt.plot(timesteps, mean_range)
plt.xlabel("Time")
plt.ylabel("Mean range")
plt.title("Mean range")

plt.figure()
plt.plot(time_periods, averages)
plt.xlabel("Time")
plt.ylabel("Average growth rate on one period")

plt.figure()
plt.plot(timesteps, mean_generation)
plt.xlabel("Time")
plt.ylabel("Mean generation")
plt.title("Mean generation")

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
plt.plot(time_10s, genetic_richness_10s)
plt.xlabel("Time")
plt.ylabel("Genetic richness")
plt.title("Genetic richness")

plt.figure()
plt.plot(timesteps, Conditions, label="Condition")
plt.plot(timesteps, mean_genotype_center, label="Mean evolution mean")
plt.xlabel("Time")
plt.title("Condition over time")
plt.legend()


plt.show()
