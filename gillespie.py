import numpy as np
import time

DISTANT_COEFF = 300


def growth_rate_exp(genes, condition):
    return np.exp(-(DISTANT_COEFF * (np.abs(genes - condition))))


def growth_rate(genes, condition):
    return 10 / (10 + DISTANT_COEFF * (np.abs(genes - condition)))


condition_profile = [(t, np.sin(2 * np.pi * t / 5)) for t in range(10000)]


def gillespie_algorithm(
    initial_evolutions, initial_populations, total_time, evolution_rate
):
    populations = np.array(initial_populations)
    time = 0
    timesteps = [time]
    population_history = [populations.copy()]
    change_time, condition = condition_profile.pop(0)
    current_rates = np.array(growth_rate(initial_evolutions, condition))
    current_evolutions = initial_evolutions
    proportions_history = [populations / sum(populations)]

    while time < total_time:
        if time > change_time and condition_profile:
            change_time, condition = condition_profile.pop(0)
            current_rates = np.array(
                [growth_rate(gene, condition) for gene in current_evolutions]
            )

        total_rate = np.sum(current_rates * populations)
        if total_rate == 0:
            break

        dt = np.random.exponential(1 / total_rate)
        time += dt

        probabilities = current_rates * populations / total_rate
        event = np.random.choice(len(populations), p=probabilities)

        populations[event] += 1
        proportions_history.append(populations / sum(populations))

        timesteps.append(time)

        evol_probability = sum(populations) * dt / (evolution_rate * total_time)
        if np.random.uniform(0, 1) < evol_probability and len(populations) < 10:
            evolving_population = np.random.choice(len(populations))
            new_evolution = np.random.normal(
                current_evolutions[evolving_population], 0.05
            )
            current_evolutions = np.append(current_evolutions, new_evolution)
            populations = np.append(populations, 1)
            current_rates = np.append(
                current_rates, growth_rate(new_evolution, condition)
            )
            population_history = [np.append(step, 0) for step in population_history]
            proportions_history = [
                np.append(step, -0.1) for step in proportions_history
            ]
        population_history.append(populations.copy())

    return timesteps, population_history, proportions_history, current_evolutions


# Example usage

initial_evolutions = np.array([-0.5, 0.7])
initial_populations = [10, 10]
total_time = 150
# np.random.seed(0)
start = time.time()
timesteps, population_history, proportions_history, current_evolutions = (
    gillespie_algorithm(initial_evolutions, initial_populations, total_time, 1000)
)
stop = time.time()

print(f"Execution time: {stop - start}s")
print(f"variant number : {len(current_evolutions)}")
print(current_evolutions)
print(proportions_history[-1])
best_gene = np.argmin(abs(current_evolutions - 0.5))
print("Rank and value of the best gene :", best_gene, current_evolutions[best_gene])
print("Proportion of this gene :", proportions_history[-1][best_gene])
# Plot the results
import matplotlib.pyplot as plt

plt.figure()
plt.plot(timesteps, population_history)
plt.xlabel("Time")
plt.ylabel("Population")
plt.legend([f"Population {i}" for i in range(len(proportions_history[0]))])

plt.figure()
plt.plot(timesteps, proportions_history)
plt.xlabel("Time")
plt.ylabel("Proportion")
plt.legend([f"Population {i}" for i in range(len(proportions_history[0]))])


plt.show()
