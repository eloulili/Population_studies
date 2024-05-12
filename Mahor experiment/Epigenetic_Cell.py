from typing import Optional
import numpy as np
import matplotlib.scale as scle
import matplotlib.pyplot as plt
import cProfile
import time

np.random.seed(0)
N = 500
probabilities = [0.1, 0.2] # evolution, loss of adaptation
first_evolution = [7]
numbers = [N]
conditions_profile = [
    (3,5000)
]
COND_COEF = 10
growth_rate_error = 0.00001

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


def inherent_growth_rate_function(condition):
    return np.exp(-condition/COND_COEF)  # assume that condition 0 is the best and condition 9 is the worst


DISTANT_COEFF = 0.5
CST_VALUE = 0.05
dt = 1
smooth_coefficient = 6000


def growth_rate_function(best_gene_distance, epigenetic, condition):
    if epigenetic is None:
        return CST_VALUE
    return CST_VALUE + np.log( max(1,1+DISTANT_COEFF * epigenetic) ) /5


    
class EvolutiveCells1D:
    def __init__(self, type: int, epigenetic: Optional[float] = None, conditions=0, growth_rate_error=0.,  growth_rate: Optional[float] = None):

        self.type = type
        self.epigenetic = epigenetic
        if growth_rate is None and epigenetic is not None:
            self.growth_rate = growth_rate_function(
                abs(epigenetic - conditions), self.epigenetic, conditions
            ) + growth_rate_error
        elif growth_rate is None:
            self.growth_rate = CST_VALUE + growth_rate_error        
        else:
            self.growth_rate = growth_rate + growth_rate_error
        self.gr_error = growth_rate_error
        self.absolute_generation = 0
        self.generation_since_last_mutation = 0
        

    def update_cell(
        self,
        conditions,
        probability_to_adapt: float = 0.1,
        adaptation_loss_probability: float = 0.5,
        stds: tuple[float, float ,float] = (0.1, 0.05, 0.1),
    ) -> tuple[float, float]:
        new_epigenetic = None
        if self.epigenetic is not None:

            if np.random.uniform(0, 1) < adaptation_loss_probability:
                self.epigenetic = None
        if np.random.uniform(0, 1) < probability_to_adapt:
                if self.epigenetic is not None:
                    new_epigenetic = np.random.normal(self.epigenetic, stds[1])
                    self.epigenetic = new_epigenetic
                else:
                    new_epigenetic = np.random.normal(0, stds[1])
                    self.epigenetic = new_epigenetic
        best_distance = None
        if new_epigenetic is None:
            self.generation_since_last_mutation = 0
        else :
            best_distance = abs(new_epigenetic - conditions)
            self.growth_rate = growth_rate_function(best_distance, self.epigenetic, conditions) + self.gr_error
        

        

        return new_epigenetic is not None, self.epigenetic is not None

    def copy(self, conditions):
        additive_error = np.random.normal(0, growth_rate_error)
        new_cell = EvolutiveCells1D(self.type, conditions=conditions, epigenetic=self.epigenetic, growth_rate=self.growth_rate - self.gr_error, growth_rate_error=additive_error + self.gr_error)
        new_cell.absolute_generation = self.absolute_generation +1
        new_cell.generation_since_last_mutation = self.generation_since_last_mutation +1
        return new_cell

    def __str__(self):
        return f"Cell of type {self.type} has adaptation {self.epigenetic} and long evolution {self.evolution}"


class EvolutiveSample1D:

    def __init__(self, cells: list[EvolutiveCells1D], nb_types: int):
        self.cells = cells
        self.n = len(cells)
        self.nb_types = nb_types
        self.cumulative_growth_rates = []
        self.conditions = None
        self.evolution_count  : dict[float,tuple[int,int]] = dict()
        for cell in self.cells:
            was_None = False
            if cell.epigenetic is None:
                was_None = True
                cell.epigenetic = -1
            if cell.epigenetic not in self.evolution_count:
                if cell.epigenetic is not None:
                    self.evolution_count[cell.epigenetic] = [(1,0)]
            else:
                self.evolution_count[cell.epigenetic][0] = (self.evolution_count[cell.epigenetic][0][0] +1 ,0)
            if was_None:
                cell.epigenetic = None
        cumul = 0.0
        for i in range(self.n):
            cumul += self.cells[i].growth_rate
            self.cumulative_growth_rates.append(cumul)

        self.total_growth_rate = cumul
        self.sum_absolute_generation = 0
        self.sum_generation_since_last_evolution = 0

        self.sum_evolution = sum([cell.epigenetic for cell in self.cells if cell.epigenetic is not None]) - sum([cell.epigenetic == None for cell in self.cells ])


        self.list_evolutions = list(set([cell.epigenetic for cell in self.cells]))
        self.n_adaptation = 0
        self.proportion_with_epigenetic = [sum([cell.epigenetic is not None for cell in self.cells])]

        self.max_evolution = [max([-1] + [cell.epigenetic for cell in self.cells if cell.epigenetic is not None])]

    def change_conditions(self, conditions):
        self.conditions = conditions
        for cell in self.cells:
            best_distance = None
            if cell.epigenetic is not None:

                best_distance = min(best_distance, abs(cell.epigenetic - conditions))
            cell.growth_rate = growth_rate_function(
                best_distance, cell.epigenetic, conditions
            ) + cell.gr_error
        cumul = 0.0
        for i in range(self.n):
            cumul += self.cells[i].growth_rate
            self.cumulative_growth_rates[i]
        self.total_growth_rate = cumul

    def get_proportions_per_type(self):
        return [
            sum([cell.type == i for cell in self.cells]) / self.n
            for i in range(self.nb_types)
        ]

    def get_mean_growth_rate_function(self):
        return self.total_growth_rate/ self.n

    def get_mean_growth_rate_by_type(self):
        growth_rate_by_type = []
        for i in range(self.nb_types):
            if sum([cell.type == i for cell in self.cells]) != 0:
                growth_rate_by_type.append(
                    sum([cell.growth_rate for cell in self.cells if cell.type == i])
                    / sum([cell.type == i for cell in self.cells])
                )
            else:
                growth_rate_by_type.append(0)
        return growth_rate_by_type

    def get_cumulative_growth_rate_function(self):
        return self.cumulative_growth_rates[-1]

    def get_proportions_per_evolution(self, evolution_list):
        return [
            sum([cell.evolution == e for cell in self.cells]) / self.n
            for e in evolution_list
        ]



    def update(
        self,
        birth_index,
        death_index,
        n_timesteps,
        adaptation_probability: float = 0.1,
        adaptation_loss_probability: float = 0.5,
    ):
        new_cell = self.cells[birth_index].copy(self.conditions)
        has_new_epigenetic, has_epigenetic = new_cell.update_cell(
            self.conditions,
            probability_to_adapt=adaptation_probability,
            adaptation_loss_probability=adaptation_loss_probability,
        )
        self.max_evolution.append(self.max_evolution[-1])
        self.cumulative_growth_rates[death_index:] = [self.cumulative_growth_rates[i] + new_cell.growth_rate - self.cells[death_index].growth_rate for i in range(death_index, self.n)]
        dead_cell = self.cells[death_index]
        if dead_cell.epigenetic is not None:
            self.proportion_with_epigenetic.append(self.proportion_with_epigenetic[-1] - 1)
            if dead_cell.epigenetic == self.max_evolution[-2] and sum([cell.epigenetic == self.cells[death_index].epigenetic for cell in self.cells]) == 1:
                self.max_evolution[-1] = max([-1] + [cell.epigenetic for cell in self.cells if cell.epigenetic is not None and cell.epigenetic != self.cells[death_index].epigenetic])
        else:
            self.proportion_with_epigenetic.append(self.proportion_with_epigenetic[-1])


        if has_epigenetic:
            evol_new_cell = new_cell.epigenetic 
            self.proportion_with_epigenetic[-1] += 1
        else:
            evol_new_cell = -1
        
        if self.cells[death_index].epigenetic is not None:
            evol_dead_cell = self.cells[death_index].epigenetic
        else:
            evol_dead_cell = -1
        

        self.sum_absolute_generation -= self.cells[death_index].absolute_generation
        self.sum_generation_since_last_evolution -= self.cells[death_index].generation_since_last_mutation
        self.total_growth_rate = self.cumulative_growth_rates[-1]
        if has_new_epigenetic:
            self.list_evolutions.append(evol_new_cell)
            self.evolution_count[evol_new_cell] = [(0,n_timesteps-1)]
            self.n_adaptation += 1
            if evol_new_cell > self.max_evolution[-2]:
                self.max_evolution[-1] = evol_new_cell

        self.cells[death_index] = new_cell
        self.sum_absolute_generation += new_cell.absolute_generation
        self.sum_generation_since_last_evolution += new_cell.generation_since_last_mutation

        if evol_dead_cell == evol_new_cell :
            pass
        else:
            self.evolution_count[evol_dead_cell].append((self.evolution_count[evol_dead_cell][-1][0]-1,n_timesteps))
            self.evolution_count[evol_new_cell].append((self.evolution_count[evol_new_cell][-1][0]+1,n_timesteps))
        self.sum_evolution += evol_new_cell - evol_dead_cell
        


    def __str__(self):
        string = ""
        for cell in self.cells:
            string += str(cell) + "\n"

        return string
    
    def find_top_evolutions(self, n_top=10):
        # Dictionnaire pour stocker la meilleure proportion atteinte pour chaque évolution
        best_proportion_per_evolution = {}

        # Parcourir toutes les évolutions connues dans l'historique
        for rank,(evolution, records) in enumerate(self.evolution_count.items()):
            # Trouver le record avec la proportion maximale pour cette évolution
            max_proportion = max(records, key=lambda x: x[0])[0]
            best_proportion_per_evolution[evolution] = max_proportion , rank

        # Trier les évolutions par leur meilleure proportion atteinte, décroissant
        sorted_evolutions = sorted(best_proportion_per_evolution.items(), key=lambda item: item[1][0], reverse=True)

        # Extraire les n_top premières évolutions
        top_evolutions = [evolution for evolution, proportion in sorted_evolutions[:n_top]]

        return top_evolutions


def Moran_process(
    sample: EvolutiveSample1D,
    conditions_profile: list[tuple[int, int]],
    adaptation_probability: float = 0.01,
    adaptation_loss_probability: float = 0.5,
):
    proportions_type = [sample.get_proportions_per_type()]
    growth_rate_by_type = [sample.get_mean_growth_rate_by_type()]
    growth_rates = [sample.get_mean_growth_rate_function()]
    absolute_generation = [sample.sum_absolute_generation/N]
    generation_since_last_evolution = [sample.sum_generation_since_last_evolution/N]
    mean_epigenetic = [sample.sum_evolution/N]
    current_time = 0
    timesteps = [0]
    n_timesteps = 1
    for conditions, change_time in conditions_profile:
        sample.change_conditions(conditions)
        while current_time < change_time:
            next_time = np.random.exponential(1 / sample.total_growth_rate)
            current_time += next_time
            birth_rate = np.random.uniform(0, sample.get_cumulative_growth_rate_function())
            birth_index = np.searchsorted(sample.cumulative_growth_rates, birth_rate)

            death_index = np.random.randint(sample.n)
            sample.update(
                birth_index,
                death_index,
                n_timesteps,
                adaptation_probability,
                adaptation_loss_probability,
            )

            growth_rates.append(sample.get_mean_growth_rate_function())
            timesteps.append(current_time)
            absolute_generation.append(sample.sum_absolute_generation/N)
            generation_since_last_evolution.append(sample.sum_generation_since_last_evolution/N)
            proportions_type.append(sample.get_proportions_per_type())
            mean_epigenetic.append(sample.sum_evolution/N)

            n_timesteps += 1

    print(
        f"Evolutions count:{len(sample.evolution_count.keys())} \n"
        f"Adaptations count:{sample.n_adaptation} \n"
        f"N time steps:{n_timesteps} "
    )

    return (
        timesteps,
        proportions_type,
        growth_rate_by_type,
        growth_rates,
        mean_epigenetic,
        sample.list_evolutions,
        absolute_generation,
        generation_since_last_evolution
    )


def main(
    first_evolution,
    numbers,
    conditions_profile: list[tuple[int, int]],
    probabilities: Optional[list[float]] = None,
    n_iter: int = 1,
):
    assert len(first_evolution) == len(numbers)
    for k in range(n_iter):
        np.random.seed(k)
        cells = []
        initial_conditions = conditions_profile[0][0]
        for i in range(len(first_evolution)):
            for j in range(numbers[i]):
                cells.append(
                    EvolutiveCells1D(i, conditions=initial_conditions)
                )
        sample = EvolutiveSample1D(
            cells,
            len(first_evolution),
        )
        start = time.time()
        if probabilities != None:
            (
                timesteps,
                proportion_adaptation,
                growth_rate_by_type,
                mean_growth_rates,
                mean_epigenetic,
                list_evolution,
                absolute_generation,
                generation_since_last_evolution
            ) = Moran_process(
                sample,
                conditions_profile,
                adaptation_probability=probabilities[0],
                adaptation_loss_probability=probabilities[1],
            )
        else:
            (
                timesteps,
                proportion_adaptation,
                growth_rate_by_type,
                mean_growth_rates,
                mean_epigenetic,
                list_evolution,
                absolute_generation,
                generation_since_last_evolution
            ) = Moran_process(sample, conditions_profile)

        print(f"Time elapsed: {time.time()-start} \n")

        start = time.time()
        top_10_evolutions = sample.find_top_evolutions( n_top=10)

        
        smoothed_growth_rates = smooth_data(mean_growth_rates, timesteps, smooth_coefficient=smooth_coefficient)
        smoothed_max_evolution = smooth_data(sample.max_evolution, timesteps, smooth_coefficient=smooth_coefficient)
        smoothed_proportion_with_epigenetics = smooth_data(sample.proportion_with_epigenetic, timesteps, smooth_coefficient=smooth_coefficient)
        smoothed_mean_epigenetic = smooth_data(mean_epigenetic, timesteps, smooth_coefficient=smooth_coefficient)
        print(f"Post process time: {time.time()-start} \n")
        plt.figure()
       
        for evolution in top_10_evolutions:
            proportion_evolution =  []
            first_appearance = sample.evolution_count[evolution][0][1]
            last_appearance = sample.evolution_count[evolution][-1][1]
            previous_change=first_appearance
            for n, t_change in sample.evolution_count[evolution]:
                proportion_evolution.extend([n]*(t_change-previous_change))
                previous_change = t_change
            plt.plot(timesteps[first_appearance:last_appearance], proportion_evolution ,label=f"Evolution {evolution}")
        plt.xlabel("Time")
        plt.ylabel("Proportion of cells")
        plt.title("Proportion of cells per evolution")
        plt.legend()

        plt.figure()
        plt.plot(timesteps, mean_growth_rates, label="Mean growth rate")
        plt.plot(timesteps[smooth_coefficient:], smoothed_growth_rates, label="Smoothed mean growth rate")
        plt.xlabel("Time")
        plt.ylabel("Mean growth rate")
        plt.title("Mean growth rate")

        plt.figure()
        plt.plot(timesteps, absolute_generation, label="Mean absolute generation")
        plt.xlabel("Time")
        plt.ylabel("Mean absolute generation")
        plt.title("Mean absolute generation")

        plt.figure()
        plt.plot(timesteps, generation_since_last_evolution, label="Mean generation since last evolution")
        plt.xlabel("Time")
        plt.ylabel("Mean generation since last evolution")
        plt.title("Mean generation since last evolution")

        plt.figure()
        plt.plot(timesteps, mean_epigenetic, label="Mean epigenetic")
        plt.plot(timesteps[smooth_coefficient:], smoothed_mean_epigenetic, label="Smoothed mean epigenetic")
        plt.xlabel("Time")
        plt.ylabel("Mean epigenetic")
        plt.title("Mean epigenetic")

        plt.figure()
        plt.plot(timesteps, sample.proportion_with_epigenetic, label="Proportion with epigenetic")
        plt.plot(timesteps[smooth_coefficient:], smoothed_proportion_with_epigenetics, label="Smoothed proportion with epigenetic")
        plt.xlabel("Time")
        plt.ylabel("Proportion with epigenetic")
        plt.title("Proportion with epigenetic")

        plt.figure()
        plt.plot(timesteps, sample.max_evolution, label="Max evolution")
        plt.plot(timesteps[smooth_coefficient:], smoothed_max_evolution, label="Smoothed max evolution")
        plt.xlabel("Time")
        plt.ylabel("Max evolution")
        plt.title("Max evolution")

        plt.figure()
        plt.plot(absolute_generation, mean_growth_rates)
        plt.xlabel("Absolute generation")
        plt.ylabel("Mean growth rate")
        plt.title("Mean growth rate by absolute generation")


        plt.show()
        (       timesteps,
                proportion_evolution,
                growth_rate_by_type,
                mean_growth_rates,
                list_evolution,
                absolute_generation,
                generation_since_last_evolution,
                mean_epigenetic,
                sample
            ) = None, None, None, None, None, None, None, None, None
    return timesteps



#cProfile.run("main(first_evolution, numbers, conditions_profile, probabilities)", sort="tottime")

main(first_evolution, numbers, conditions_profile, probabilities)

