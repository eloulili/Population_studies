import numpy as np

DISTANT_COEFF = 5

def growth_rate(gene, condition):
    return  np.exp(-(DISTANT_COEFF*abs(gene-condition)))

condition_profile = [(t, abs(np.sin(t/3.14))) for t in range(10)]

def gillespie_algorithm(initial_evolutions, initial_populations, total_time, evolution_rate):
    populations = np.array(initial_populations)
    time = 0
    timesteps = [time]
    population_history = [populations.copy()]
    change_time, condition = condition_profile.pop(0)
    current_rates = np.array([growth_rate(gene, condition) for gene in initial_evolutions])
    current_evolutions = initial_evolutions
    proportions_history = [populations / sum(populations)]

    while time < total_time:
        if time > change_time and condition_profile:
            change_time, condition = condition_profile.pop(0)
            current_rates = np.array([growth_rate(gene, condition) for gene in current_evolutions])

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


        evol_probability = np.exp(-evolution_rate /(sum(populations*dt)))
        if np.random.uniform(0,1) < evol_probability:
            evolving_population = np.random.choice(len(populations))
            new_evolution = np.random.normal(current_evolutions[evolving_population], 0.05)
            current_evolutions = np.append(current_evolutions, new_evolution)
            populations = np.append(populations, 1)
            current_rates = np.append(current_rates, growth_rate(new_evolution, condition))
            for time_step in range(len(population_history)):
                population_history[time_step] = np.append(population_history[time_step], 0)
            for time_step in range(len(proportions_history)):
                proportions_history[time_step] = np.append(proportions_history[time_step], 0.)
        population_history.append(populations.copy())



    return timesteps, population_history, proportions_history

# Example usage
N = 2
initial_evolutions = np.array([0.3, 0.5])
initial_populations = [10, 10]
total_time = 10
timesteps, population_history, proportions_history = gillespie_algorithm( initial_evolutions, initial_populations, total_time, 50)

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
