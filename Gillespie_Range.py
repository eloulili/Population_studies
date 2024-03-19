from typing import Optional
import numpy as np
import time
import matplotlib.pyplot as plt

DISTANT_COEFF = 5
RANGE_COEFF = 1.5

np.random.seed(0)

def distant_range(min,max, e):
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
        min_point = int((min_g - min)/(max - min)*num_point)
        max_point = int((max_g - min)/(max - min)*num_point)
        for i in range(min_point, max_point):
            value[i] += 1
    value = [v/len(genotypes) for v in value]

    plt.plot(points, value)
    plt.xlabel('Valeurs')
    plt.ylabel('Densité de probabilité')
    plt.title('Distribution des valeurs'+ " population: "+ str(len(genotypes)))
    plt.grid(True)
    plt.show()

def growth_rate( genotype, phenotype, condition):
    min_g, max_g = genotype
    mini, maxi = min_g, max_g
    if phenotype is not None:
        min_p, max_p = phenotype
        mini, maxi = min(min_p, min_g), max(max_p, max_g)
    return 1/((1 + np.abs(max_g - min_g)*RANGE_COEFF)) * np.exp(-DISTANT_COEFF*(distant_range(mini, maxi, condition))) # Penality for the distance and the range



class EvolutiveCells1D:
    def __init__(self, first_evolution: tuple[float, float], conditions = 0):

        self.phenotype = None # the first element is the adaptation, the second is the duration
        self.genotype = first_evolution
        self.growth_rate = growth_rate(first_evolution,None ,conditions)
        self.generation = 0

    
    
    def copy(self, conditions):
        new_cell = EvolutiveCells1D(self.genotype, conditions)
        new_cell.phenotype = self.phenotype
        new_cell.genotype = self.genotype
        new_cell.generation = self.generation + 1
        return new_cell




def gillespie_algorithm(initial_evolutions, 
                        total_time, 
                        condition_profile, 
                        n_plot=5, 
                        evolutions_probability=0.1, 
                        adaptation_probability=0.1, 
                        evol_probability_without_adaptation=0.1,
                        probability_to_loose_adaptation=1/5, 
                        death_rate=0.5):
    populations = np.array([len(initial_evolutions)])
    n_evol, n_adapt, n_evol_wa = 0, 0, 0
    n_events, n_death, n_born = 0 , 0, 0
    time = 0
    timesteps = [time]
    change_time, condition = condition_profile.pop(0)
    time_between_plot = total_time/n_plot
    time_next_plot = time_between_plot
    current_evolutions = list(initial_evolutions)
    print_hist(current_evolutions)
    cell_batch = [EvolutiveCells1D(evolution, condition) for evolution in initial_evolutions]
    current_rates = np.array([cell.growth_rate for cell in cell_batch])
    current_rates = np.insert(current_rates, 0, death_rate * len(cell_batch))
    rate_evolution = [ np.mean(current_rates[1:])]
    mean_range = [np.mean([genotype[1] - genotype[0] for genotype in current_evolutions])]
    
    while time < total_time:
        if time > change_time and condition_profile:
            change_time, condition = condition_profile.pop(0)
            current_rates = np.array([growth_rate(cell.genotype, cell.phenotype, condition) for cell in cell_batch])
            current_rates = np.insert(current_rates, 0, death_rate * len(cell_batch))

        
        if len(cell_batch) == 0:
            print("extinction")
            break

        if len(cell_batch) > 13000:
            print("overpopulation")
            generations = [cell.generation for cell in cell_batch]
            print(f"N_evolution : {n_evol}, N_adaptation : {n_adapt}, N_death : {n_death}, N_born : {n_born}, N_events : {n_events}, N_evol_without_adaptation : {n_evol_wa}")
           # print(f"Mean generation : {np.mean(generations)}, Max generation : {max(generations)}, Min generation : {min(generations)}, Median generation : {np.median(generations)} , Std generation : {np.std(generations)}")
            print_hist(current_evolutions)
            break
        total_rate = np.sum(current_rates)
        dt = np.random.exponential(1 / total_rate)
        time += dt

        probabilities = current_rates / total_rate
        event = np.random.choice(len(current_rates), p=probabilities)
        n_events += 1

        timesteps.append(time)

        if event == 0: #death
            dead_cell = np.random.choice(len(cell_batch))
            cell_batch.pop(dead_cell)
            current_rates = np.delete(current_rates, dead_cell+1)
            current_rates[0] -= death_rate
            current_evolutions.pop(dead_cell)
            n_death += 1
            populations = np.append(populations, len(cell_batch))
        else:
            new_cell = cell_batch[event-1].copy(condition)
            cell_batch.append(new_cell)
            n_born += 1
            populations = np.append(populations, len(cell_batch))
            current_rates[0] += death_rate

            change_probabilities = np.random.uniform(0,1,4)

            if change_probabilities[0] < evol_probability_without_adaptation:
                new_cell.genotype = np.random.normal(new_cell.genotype[0], 0.05), np.random.normal(new_cell.genotype[1], 0.05)
                n_evol += 1
                n_evol_wa += 1

            if change_probabilities[1] < adaptation_probability:
                new_cell.phenotype = np.random.normal(new_cell.genotype[0], 0.1), np.random.normal(new_cell.genotype[1], 0.1)
                n_adapt += 1

            if change_probabilities[2] < probability_to_loose_adaptation:
                new_cell.phenotype = None

            if change_probabilities[3] < evolutions_probability and new_cell.phenotype is not None:
                mini, maxi = np.random.normal(new_cell.phenotype[0], 0.05), np.random.normal(new_cell.phenotype[1], 0.05)
                if mini < maxi:
                    new_cell.genotype = mini, maxi
                else:
                    new_cell.genotype = maxi, mini
                n_evol += 1

            current_evolutions.append(new_cell.genotype)
            current_rates = np.append(current_rates, growth_rate(new_cell.genotype, new_cell.phenotype, condition))

        rate_evolution.append(np.mean(current_rates[1:]))
        mean_range.append(np.mean([genotype[1] - genotype[0] for genotype in current_evolutions]))
        
        if time > time_next_plot:
            generations = [cell.generation for cell in cell_batch]
            print(f"N_evolution : {n_evol}, N_adaptation : {n_adapt}, N_death : {n_death}, N_born : {n_born}, N_events : {n_events}, N_evol_without_adaptation : {n_evol_wa}")
            print(f"Mean generation : {np.mean(generations)}, Max generation : {max(generations)}, Min generation : {min(generations)}, Median generation : {np.median(generations)} , Std generation : {np.std(generations)}")
            print_hist(current_evolutions)
            print(f"Time : {time}")
            time_next_plot += time_between_plot

    return timesteps, populations, current_evolutions, rate_evolution, mean_range



period_duration = 60
condition_profile = [(t/10, np.sin(2 * np.pi * t/(10*period_duration))**2) for t in range(6000)]

population = 2000
adaptation_probability = 0.1
evolution_probability = 0.1
evolution_without_adaptation_probability = 5e-4
mean_time_to_loose_adaptation = 5
probability_to_loose_adaptation = 1/mean_time_to_loose_adaptation
maxs = np.random.uniform(0.5, 1.2, population)
mins = np.random.uniform(-0.2, 0.5, population)
initial_evolutions = [(mins[i], maxs[i]) for i in range(population)]
death_rate = 0.4
total_time =600
start = time.time()
timesteps,  populations, current_evolutions, rate_evolution, mean_range = gillespie_algorithm(  initial_evolutions,  
                                                                                                total_time, 
                                                                                                condition_profile,
                                                                                                evolutions_probability = evolution_probability,
                                                                                                adaptation_probability = adaptation_probability,
                                                                                                evol_probability_without_adaptation = evolution_without_adaptation_probability, 
                                                                                                probability_to_loose_adaptation = probability_to_loose_adaptation,
                                                                                                death_rate=death_rate,
                                                                                                n_plot=1)
stop = time.time()



print(f"Execution time (gilespie algorithm): {stop - start}s")
print(f"timesteps : {len(timesteps)}")

start = time.time()
normalized_growth_rates = [rate_evolution[i]*(timesteps[i+1]-timesteps[i]) for i in range(len(rate_evolution)-1)]
averages = np.array([])
end_indexes = []   
time_periods = []
end_index = 0
for start_index in range(len(timesteps)-1):
        while not( timesteps[end_index] - timesteps[start_index] >= period_duration):
            end_index += 1
            if end_index == len(timesteps):
                break
        if end_index == len(timesteps):
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
averages = np.diff(cumulative_growth_sum[end_indexes] - cumulative_growth_sum[range(len(end_indexes))])

# Calculez la moyenne du taux de croissance pour chaque période de temps en divisant les sommes cumulatives par la durée de la période
averages = averages / period_duration
stop = time.time()
print(f"Execution time (average calculation): {stop - start}s")
# Plot the results

plt.figure()
plt.plot(timesteps, populations)
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Population evolution')


plt.figure()
plt.plot(timesteps, rate_evolution)
plt.xlabel('Time')
plt.ylabel('Rate evolution')
plt.title('Rate evolution')

plt.figure()
plt.plot(timesteps, mean_range)
plt.xlabel('Time')
plt.ylabel('Mean range')
plt.title('Mean range')

plt.figure()
plt.plot(time_periods, averages)
plt.xlabel('Time')
plt.ylabel('Average growth rate')


plt.show()
