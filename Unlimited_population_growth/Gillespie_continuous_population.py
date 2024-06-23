from typing import Optional
import numpy as np
import time
import matplotlib.pyplot as plt

DISTANT_COEFF = 250

np.random.seed(0)


def growth_rate(genes, adaptation, condition):
    if adaptation is None:
        return 0.8 * 10 / (10 + DISTANT_COEFF * (np.abs(genes - condition)))
    return 0.8 * max(
        10 / (10 + DISTANT_COEFF * (np.abs(genes - condition))),
        10 / (10 + DISTANT_COEFF * (np.abs(adaptation - condition))),
    )


class EvolutiveCells1D:
    def __init__(self, first_evolution: Optional[int] = None, conditions=0):

        self.short_evolution = (
            None  # the first element is the adaptation, the second is the duration
        )
        self.long_evolution = first_evolution
        self.growth_rate = growth_rate(first_evolution, None, conditions)
        self.generation = 0

    def update_cell(
        self,
        conditions,
        probability_to_evolve: float = 0.1,
        distance_mult: float = 1.1,
        probability_to_adapt: float = 0.1,
    ) -> tuple[int, int]:
        best_distance = abs(self.long_evolution - conditions)
        new_evolution = -1
        new_adaptation = -1
        if self.short_evolution != []:
            best_adaptation = min(
                self.short_evolution, key=lambda x: abs(x[0] - conditions)
            )
            best_distance = min(best_distance, abs(best_adaptation[0] - conditions))
        self.growth_rate = growth_rate(best_distance, self.long_evolution, conditions)
        for evolution in self.short_evolution:
            if evolution[1] == 0:
                self.short_evolution.remove(evolution)
            else:
                evolution[1] -= 1

            distance = abs(evolution[0] - self.long_evolution)
            if np.random.uniform(0, 1) < probability_to_evolve / (
                distance_mult**distance
            ):
                self.long_evolution = evolution[0]
                new_evolution = evolution[0]

        if (
            np.random.uniform(0, 1) < probability_to_adapt
            and self.long_evolution != conditions
        ):
            if self.long_evolution < conditions:
                new_adaptation = np.random.randint(self.long_evolution, conditions + 1)
                self.short_evolution.append([new_adaptation, 5])
            else:
                new_adaptation = np.random.randint(conditions, self.long_evolution + 1)
                self.short_evolution.append([new_adaptation, 5])
        return new_evolution, new_adaptation

    def copy(self, conditions):
        new_cell = EvolutiveCells1D( conditions)
        new_cell.short_evolution = self.short_evolution
        new_cell.long_evolution = self.long_evolution
        new_cell.generation = self.generation + 1
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
    death_rate=0.5,
):
    populations = np.array([len(initial_evolutions)])
    n_evol, n_adapt, n_evol_wa = 0, 0, 0
    n_events, n_death, n_born = 0, 0, 0
    time = 0
    timesteps = [time]
    change_time, condition = condition_profile.pop(0)
    time_between_plot = total_time / n_plot
    time_next_plot = time_between_plot
    current_evolutions = np.array(initial_evolutions)
    num_bins = 60
    genetic_richness = np.array(
        [100 * len(set(current_evolutions)) / len(current_evolutions)]
    )
    plt.hist(current_evolutions, bins=num_bins, density=True, edgecolor="black")
    plt.xlabel("Valeurs")
    plt.ylabel("Densité de probabilité")
    plt.title("Distribution des valeurs")
    plt.grid(True)
    plt.show()
    cell_batch = [
        EvolutiveCells1D(evolution, condition) for evolution in initial_evolutions
    ]
    current_rates = np.array([cell.growth_rate for cell in cell_batch])
    current_rates = np.insert(current_rates, 0, death_rate * len(cell_batch))
    rate_evolution = [np.mean(current_rates[1:])]

    while time < total_time:
        if time > change_time and condition_profile:
            change_time, condition = condition_profile.pop(0)
            current_rates = np.array(
                [
                    growth_rate(cell.long_evolution, cell.short_evolution, condition)
                    for cell in cell_batch
                ]
            )
            current_rates = np.insert(current_rates, 0, death_rate * len(cell_batch))

        if len(cell_batch) == 0:
            print("extinction")
            break

        if len(cell_batch) > 13000:
            print("overpopulation")
            generations = [cell.generation for cell in cell_batch]
            print(
                f"N_evolution : {n_evol}, N_adaptation : {n_adapt}, N_death : {n_death}, N_born : {n_born}, N_events : {n_events}, N_evol_without_adaptation : {n_evol_wa}"
            )
            print(
                f"Mean generation : {np.mean(generations)}, Max generation : {max(generations)}, Min generation : {min(generations)}, Median generation : {np.median(generations)} , Std generation : {np.std(generations)}"
            )
            plt.hist(current_evolutions, bins=num_bins, density=True, edgecolor="black")
            plt.xlabel("Valeurs")
            plt.ylabel("Densité de probabilité")
            plt.title(
                "Distribution des valeurs au temps "
                + str(time)
                + "s"
                + "  population : "
                + str(len(cell_batch))
            )
            plt.grid(True)
            plt.show()
            break
        total_rate = np.sum(current_rates)
        dt = np.random.exponential(1 / total_rate)
        time += dt

        probabilities = current_rates / total_rate
        event = np.random.choice(len(current_rates), p=probabilities)
        n_events += 1

        timesteps.append(time)

        if event == 0:  # death
            dead_cell = np.random.choice(len(cell_batch))
            cell_batch.pop(dead_cell)
            current_rates = np.delete(current_rates, dead_cell + 1)
            current_rates[0] -= death_rate
            current_evolutions = np.delete(current_evolutions, dead_cell)
            n_death += 1
            populations = np.append(populations, len(cell_batch))
        else:
            new_cell = cell_batch[event - 1].copy(condition)
            cell_batch.append(new_cell)
            n_born += 1
            populations = np.append(populations, len(cell_batch))
            current_rates[0] += death_rate

            change_probabilities = np.random.uniform(0, 1, 4)

            if change_probabilities[0] < evol_probability_without_adaptation:
                new_evolution = np.random.normal(new_cell.long_evolution, 0.05)
                n_evol += 1
                n_evol_wa += 1

            if change_probabilities[1] < adaptation_probability:
                new_cell.short_evolution = np.random.normal(
                    new_cell.long_evolution, 0.1
                )
                n_adapt += 1

            if change_probabilities[2] < probability_to_loose_adaptation:
                new_cell.short_evolution = None

            if (
                change_probabilities[3] < evolutions_probability
                and new_cell.short_evolution is not None
            ):
                new_evolution = np.random.normal(new_cell.short_evolution, 0.05)
                new_cell.long_evolution = new_evolution
                current_evolutions = np.append(current_evolutions, new_evolution)
                n_evol += 1

            current_evolutions = np.append(current_evolutions, new_cell.long_evolution)
            current_rates = np.append(
                current_rates,
                growth_rate(
                    new_cell.long_evolution, new_cell.short_evolution, condition
                ),
            )

        rate_evolution.append(np.mean(current_rates[1:]))
        genetic_richness = np.append(
            genetic_richness,
            100 * len(set(current_evolutions)) / len(current_evolutions),
        )

        if time > time_next_plot:
            generations = [cell.generation for cell in cell_batch]
            print(
                f"N_evolution : {n_evol}, N_adaptation : {n_adapt}, N_death : {n_death}, N_born : {n_born}, N_events : {n_events}, N_evol_without_adaptation : {n_evol_wa}"
            )
            print(
                f"Mean generation : {np.mean(generations)}, Max generation : {max(generations)}, Min generation : {min(generations)}, Median generation : {np.median(generations)} , Std generation : {np.std(generations)}"
            )
            plt.hist(current_evolutions, bins=num_bins, density=True, edgecolor="black")
            plt.xlabel("Valeurs")
            plt.ylabel("Densité de probabilité")
            plt.title(
                "Distribution des valeurs au temps "
                + str(time)
                + "s"
                + "  population : "
                + str(len(cell_batch))
            )
            plt.grid(True)
            plt.show()
            time_next_plot += time_between_plot

    return timesteps, populations, current_evolutions, rate_evolution, genetic_richness


basic_condition_profile = [
    (1, 0.9),
    (3, 0.97),
    (5, 1.05),
    (8, 1.1),
    (12, 1.15),
    (14, 1.2),
    (16, 1.25),
    (18, 1.35),
    (20, 1.45),
    (35, 1.5),
    (40, 1.2),
]

condition_profile = []
for i in range(15):
    for times, condition in basic_condition_profile:
        condition_profile.append((times + i * 40, condition))

condition_profile = [(15, 1.15), (90, 1.3), (110, 1.2)]

population = 6000
adaptation_probability = 0.1
evolution_probability = 0.1
evolution_without_adaptation_probability = 5e-4
mean_time_to_loose_adaptation = 5
probability_to_loose_adaptation = 1 / mean_time_to_loose_adaptation
initial_evolutions = np.random.normal(0, 0.2, population)
death_rate = 0.52
total_time = 300
start = time.time()
timesteps, populations, current_evolutions, rate_evolution, genetic_richness = (
    gillespie_algorithm(
        initial_evolutions,
        total_time,
        condition_profile,
        evolutions_probability=evolution_probability,
        adaptation_probability=adaptation_probability,
        evol_probability_without_adaptation=evolution_without_adaptation_probability,
        probability_to_loose_adaptation=probability_to_loose_adaptation,
        death_rate=death_rate,
        n_plot=2,
    )
)
stop = time.time()

timesteps_10s = [timesteps[i] for i in range(len(timesteps)) if timesteps[i] > 10]
ignored_timesteps = len(timesteps) - len(timesteps_10s)
richness_10s = genetic_richness[ignored_timesteps:]

print(f"Execution time: {stop - start}s")

# Plot the results

plt.figure()
plt.plot(timesteps, populations)
plt.xlabel("Time")
plt.ylabel("Population")


plt.figure()
plt.plot(timesteps, rate_evolution)
plt.xlabel("Time")
plt.ylabel("Rate evolution")

plt.figure()
plt.plot(timesteps_10s, richness_10s)
plt.xlabel("Time")
plt.ylabel("Genetic richness")


plt.show()
"""
print(f"Execution time: {stop - start}s")
print(f"variant number : {len(current_evolutions)}")
print(current_evolutions)
print(proportions_history[-1])
best_gene = np.argmin(abs(current_evolutions - 0.5))
print("Rank and value of the best gene :" , best_gene, current_evolutions[best_gene])
print("Proportion of this gene :",proportions_history[-1][best_gene])
# Plot the results
import matplotlib.pyplot as plt
plt.figure()
plt.plot(timesteps, population_history)
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend([f'Population {i}' for i in range(len(proportions_history[0]))])

plt.figure()
plt.plot(timesteps, proportions_history)
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.legend([f'Population {i}' for i in range(len(proportions_history[0]))])



plt.show()
"""
