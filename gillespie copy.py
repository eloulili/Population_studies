from typing import Optional
import numpy as np
import time
import matplotlib.pyplot as plt

DISTANT_COEFF = 120

def growth_rate(genes, adaptation, condition):
    if adaptation is None:
        return 10/(10+DISTANT_COEFF * (np.abs(genes - condition)))
    return max( 10/(10+DISTANT_COEFF * (np.abs(genes - condition))), 10/(10+DISTANT_COEFF * (np.abs(adaptation - condition))))


class EvolutiveCells1D:
    def __init__(self, type:int, first_evolution: Optional[int] = None, conditions = 0):

        self.type = type
        self.short_evolution = None # the first element is the adaptation, the second is the duration
        self.long_evolution = first_evolution
        self.growth_rate = growth_rate(first_evolution,None ,conditions)

    def update_cell(self, conditions, probability_to_evolve: float = 0.1, distance_mult:float = 1.1 ,probability_to_adapt: float = 0.1)->tuple[int,int]:
        best_distance = abs(self.long_evolution-conditions)
        new_evolution = -1
        new_adaptation = -1
        if self.short_evolution != []:
            best_adaptation = min(self.short_evolution, key=lambda x:abs(x[0]-conditions))
            best_distance = min(best_distance, abs(best_adaptation[0]-conditions))
        self.growth_rate = growth_rate(best_distance, self.long_evolution, conditions)
        for evolution in self.short_evolution:
            if evolution[1] == 0:
                self.short_evolution.remove(evolution)
            else:
               evolution[1] -= 1

            distance = abs(evolution[0]-self.long_evolution)
            if np.random.uniform(0,1) < probability_to_evolve/(distance_mult**distance):
                self.long_evolution = evolution[0]
                new_evolution = evolution[0]

        if np.random.uniform(0,1) < probability_to_adapt and self.long_evolution != conditions:
                if self.long_evolution < conditions:
                    new_adaptation = np.random.randint(self.long_evolution,conditions+1)
                    self.short_evolution.append([new_adaptation, 5])
                else:
                    new_adaptation = np.random.randint(conditions,self.long_evolution+1)
                    self.short_evolution.append([new_adaptation, 5])
        return new_evolution, new_adaptation
    
    def copy(self, conditions):
        new_cell = EvolutiveCells1D(self.type,conditions)
        new_cell.short_evolution = self.short_evolution
        new_cell.long_evolution = self.long_evolution
        return new_cell




def gillespie_algorithm(initial_evolutions, total_time, condition_profile, n_plot = 5, evolutions_probability = 0.1, adaptation_probability = 0.1, evol_probability_without_adaptation = 0.1, probability_to_loose_adaptation = 1/5, death_rate = 0.5) -> tuple[list[int], list[list[int]], list[list[float]], list[float]]:
    populations = [len(initial_evolutions)]
    n_evol, n_adapt = 0, 0
    n_events, n_death, n_born = 0 , 0, 0
    time = 0
    timesteps = [time]
    #population_history = [populations.copy()]
    change_time, condition = condition_profile.pop(0)
    time_between_plot = total_time/n_plot
    time_next_plot = time_between_plot
    current_evolutions = list(initial_evolutions)
    # Nombre de bins pour l'histogramme
    num_bins = 30
    genetic_richness = [len(set(current_evolutions))/len(current_evolutions)]

    # Tracer l'histogramme
    plt.hist(current_evolutions, bins=num_bins, density=True, edgecolor='black')

    # Ajouter des étiquettes et un titre
    plt.xlabel('Valeurs')
    plt.ylabel('Densité de probabilité')
    plt.title('Distribution des valeurs')

    # Afficher la grille
    plt.grid(True)

    # Afficher le graphique
    plt.show()
    cell_batch = [EvolutiveCells1D(evolution, condition) for evolution in initial_evolutions]
    current_rates = [cell.growth_rate for cell in cell_batch]
    current_rates.insert(0,death_rate * len(cell_batch))
    rate_evolution = [np.mean(current_rates[1:]) ]

    while time < total_time:
        if time > change_time and condition_profile:
            change_time, condition = condition_profile.pop(0)
            current_rates = [growth_rate(cell.long_evolution, cell.short_evolution, condition) for cell in cell_batch]
            current_rates.insert(0,death_rate  * len(cell_batch))

        total_rate = np.sum(current_rates)
        if len(cell_batch) == 0:
            print("extinction")
            break

        dt = np.random.exponential(1 / total_rate)
        time += dt

        probabilities = current_rates / total_rate
        event = np.random.choice(len(current_rates), p=probabilities)
        n_events += 1

        timesteps.append(time)


        if event == 0: #death
            dead_cell = np.random.choice(len(cell_batch))
            cell_batch.pop(dead_cell)
            current_rates.pop(dead_cell+1)
            current_rates[0] -= death_rate
            current_evolutions.pop(dead_cell)
            n_death += 1
            populations.append(len(cell_batch))



        else :
            new_cell = cell_batch[event-1].copy(condition)
            cell_batch.append(new_cell)
            n_born += 1
            populations.append(len(cell_batch))
            current_rates[0] += death_rate


            change_probabilities = np.random.uniform(0,1,4)

            if change_probabilities[0] < evol_probability_without_adaptation:
                cell_batch[-1].long_evolution = np.random.normal(cell_batch[-1].long_evolution, 0.05)
                current_evolutions.append(cell_batch[-1].long_evolution)
                n_evol += 1

            if change_probabilities[1] < adaptation_probability:
                cell_batch[-1].short_evolution = np.random.normal(cell_batch[-1].long_evolution, 0.1)
                n_adapt += 1

            if change_probabilities[2] < probability_to_loose_adaptation:
                cell_batch[-1].short_evolution = None

            if change_probabilities[3] < evolutions_probability and cell_batch[-1].short_evolution != None:
                cell_batch[-1].long_evolution = np.random.normal(cell_batch[-1].short_evolution, 0.05)
                current_evolutions.append(cell_batch[-1].long_evolution)
                n_evol += 1


            current_evolutions.append(cell_batch[-1].long_evolution)
            current_rates.append(growth_rate(cell_batch[-1].long_evolution, cell_batch[-1].short_evolution, condition))

        rate_evolution.append(np.mean(current_rates[1:]))
        genetic_richness.append(len(set(current_evolutions))/len(current_evolutions))
        if time > time_next_plot:
            print(f"N_evolution : {n_evol}, N_adaptation : {n_adapt}, N_death : {n_death}, N_born : {n_born}, N_events : {n_events}")
            # Tracer l'histogramme
            plt.hist(current_evolutions, bins=num_bins, density=True, edgecolor='black')

            # Ajouter des étiquettes et un titre
            plt.xlabel('Valeurs')
            plt.ylabel('Densité de probabilité')
            plt.title('Distribution des valeurs au temps ' + str(time) + 's' + "  population : " + str(len(cell_batch)))

            # Afficher la grille
            plt.grid(True)

            # Afficher le graphique
            plt.show()
            time_next_plot += time_between_plot



    return timesteps, populations, current_evolutions, rate_evolution, genetic_richness
# Example usage

condition_profile = [(1,0.9), (3,0.97), (5, 1.05), (8,1.1), (12, 1.15), (14, 1.2), (16, 1.25), (18, 1.35), (20, 1.45),(35, 1.5), (40,1.2), (41,0.9), (43,0.97),(45, 1.05), (48,1.1), (52, 1.15), (54, 1.2), (56, 1.25), (58, 1.35), (60, 1.45),(75, 1.5), (80,1.2), (81,0.9), (83,0.97),(85, 1.05), (88,1.1), (92, 1.15), (94, 1.2), (96, 1.25), (98, 1.35), (100, 1.45),(115, 1.5), (120,1.2)]
population =7000
adaptation_probability = 0.1
evolution_probability = 0.1
evolution_without_adaptation_probability = 1e-3
mean_time_to_loose_adaptation = 5
probability_to_loose_adaptation = 1/mean_time_to_loose_adaptation
initial_evolutions = np.random.normal(0,1, population)
death_rate = 0.41
total_time = 120
#np.random.seed(0)
start = time.time()
timesteps,  populations, current_evolutions, rate_evolution, genetic_richness = gillespie_algorithm( initial_evolutions,  total_time, condition_profile, evolutions_probability = evolution_probability, adaptation_probability = adaptation_probability, evol_probability_without_adaptation = evolution_without_adaptation_probability, probability_to_loose_adaptation = probability_to_loose_adaptation, death_rate=death_rate, n_plot=10)
stop = time.time()

timesteps_5s = [timesteps[i] for i in range(len(timesteps)) if timesteps[i] > 5]
ignored_timesteps = len(timesteps) - len(timesteps_5s)
richness_5s = genetic_richness[ignored_timesteps:]

print(f"Execution time: {stop - start}s")

# Plot the results

plt.figure()
plt.plot(timesteps, populations)
plt.xlabel('Time')
plt.ylabel('Population')


plt.figure()
plt.plot(timesteps, rate_evolution)
plt.xlabel('Time')
plt.ylabel('Rate evolution')

plt.figure()
plt.plot(timesteps_5s, richness_5s)
plt.xlabel('Time')
plt.ylabel('Genetic richness')



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