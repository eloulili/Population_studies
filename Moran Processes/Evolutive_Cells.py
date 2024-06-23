from typing import Optional
import numpy as np
import matplotlib.scale as scle
import matplotlib.pyplot as plt
import cProfile
import time

np.random.seed(0)

"""
In this simulation, we consider a population of cells that can evolve and adapt to their environment.
The cells are characterized by their type, their evolution and their epigenetic adaptation.
The growth rate of a cell is determined by the distance between its evolution and the conditions of the environment.
An evolution is a long-term change in the cell's growth rate, while an adaptation is a short-term change.
"""


N = 250
probabilities = [0.1, 0.001, 0.00005] # evolution, adaptation, evolution without adaptation
first_evolution = [7]
numbers = [N]
conditions_profile = [
    (5,40000)
]
COND_COEF = 10
growth_rate_error = 0.00002

def inherent_growth_rate_function(condition):
    return np.exp(-condition/COND_COEF)  # assume that condition 0 is the best and condition 9 is the worst


DISTANT_COEFF = 1
CST_VALUE = 0.05

"""
    Define the disistribution of the next time step and the growth rate function based on
    the distance between the current evolution and the conditions
"""

def compute_next_time_step(n: int, base_growth_rate: float, type:str, variance:float = 0.05) -> float:
    if type == "lognormal":
        sigma = np.sqrt(np.log(1 + variance / (base_growth_rate * base_growth_rate)))
        mu = np.log(base_growth_rate) - sigma ** 2 / 2
        return np.random.lognormal(mu , sigma)
    if type == "exponential":
        return np.random.exponential(1 / (n * base_growth_rate))


def growth_rate_function(best_gene_distance, evolutions, condition):
    return (
        1 / (1 + 0.1 * best_gene_distance)
        + inherent_growth_rate_function(max(evolutions, condition))
    )
    # the first term is the adaptation, the second term is the inherent growth rate that can be capted by the cell


def growth_rate_function(best_gene_distance, evolutions, condition):
    return np.exp(
        -(DISTANT_COEFF * best_gene_distance)
       
    )

def growth_rate_function(best_gene_distance, evolutions, condition):
    if evolutions is None:
        return CST_VALUE
    return CST_VALUE + 0.2 * np.exp( -(DISTANT_COEFF * best_gene_distance))

#def growth_rate_function(best_gene_distance, evolutions, condition):
#    return CST_VALUE

def growth_rate_function(best_gene_distance, evolutions, condition):
    return CST_VALUE + 0.2 * np.exp(
        -(DISTANT_COEFF * best_gene_distance)
    )

class EvolutiveCells1D:
    def __init__(self, type: int,first_evolution: Optional[int] = None, conditions=0, growth_rate_error=0.,  growth_rate: Optional[float] = None):

        self.type = type
        self.epigenetic = None
        self.evolution = first_evolution
        if growth_rate is None:
            self.growth_rate = growth_rate_function(
                abs(first_evolution - conditions), self.evolution, conditions
            ) + growth_rate_error
        else:
            self.growth_rate = growth_rate + growth_rate_error
        self.gr_error = growth_rate_error
        self.absolute_generation = 0
        self.generation_since_last_evolution = 0
        

    def get_growth_rate_function(self):
        return self.growth_rate + self.gr_error

    def get_name(self):
        return self.name

    def update_cell(
        self,
        conditions,
        probability_to_evolve: float = 0.1,
        distance_mult: float = 1.1,
        probability_to_adapt: float = 0.1,
        probability_to_evolve_without_adapt: float = 0.001,
        adaptation_loss_probability: float = 0.5,
        std: float = 0.1,
    ) -> tuple[float, float]:
        new_evolution = -1
        new_adaptation = -1
        if self.epigenetic is not None:
           
            if np.random.uniform(0, 1) < probability_to_evolve :
                self.evolution = self.epigenetic + np.random.normal(0, std)
                new_evolution = self.evolution

            if np.random.uniform(0, 1) < adaptation_loss_probability:
                self.epigenetic = None


        if (
            np.random.uniform(0, 1) < probability_to_adapt
            and self.evolution != conditions
        ):
                new_adaptation = np.random.normal(self.evolution, std)
                self.epigenetic = new_adaptation
        if np.random.uniform(0, 1) < probability_to_evolve_without_adapt:
            self.evolution = self.evolution + np.random.normal(0, std)
            new_evolution = self.evolution
        best_distance = abs(self.evolution - conditions)
        if self.epigenetic is not None:
            best_distance = min(best_distance, abs(self.epigenetic - conditions))
        self.growth_rate = growth_rate_function(best_distance, self.evolution, conditions) + self.gr_error

        return new_evolution, new_adaptation

    def copy(self, conditions):
        additive_error = np.random.normal(0, growth_rate_error)
        new_cell = EvolutiveCells1D(self.type, conditions=conditions, first_evolution=self.evolution, growth_rate=self.growth_rate - self.gr_error, growth_rate_error=additive_error + self.gr_error)
        new_cell.epigenetic = self.epigenetic
        new_cell.absolute_generation = self.absolute_generation +1
        new_cell.generation_since_last_evolution = self.generation_since_last_evolution +1
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
            if cell.evolution not in self.evolution_count:
                self.evolution_count[cell.evolution] = [(1,0)]
            else:
                self.evolution_count[cell.evolution][0] = (self.evolution_count[cell.evolution][0][0] +1 ,0)
        cumul = 0.0
        for i in range(self.n):
            cumul += self.cells[i].growth_rate
            self.cumulative_growth_rates.append(cumul)

        self.total_growth_rate = cumul
        self.sum_absolute_generation = 0
        self.sum_generation_since_last_evolution = 0

        self.sum_evolution = sum([cell.evolution for cell in self.cells])


        self.list_evolutions = list(set([cell.evolution for cell in self.cells]))
        self.n_adaptation = 0

    def change_conditions(self, conditions):
        self.conditions = conditions
        for cell in self.cells:
            best_distance = abs(cell.evolution - conditions)
            if cell.epigenetic is not None:

                best_distance = min(best_distance, abs(cell.epigenetic - conditions))
            cell.growth_rate = growth_rate_function(
                best_distance, cell.evolution, conditions
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
        evolution_probabilty: float = 0.01,
        adaptation_probability: float = 0.01,
        evolution_without_epigenesis: float = 0.0001,
        distance_mult: float = 1.1,

    ):
        new_cell = self.cells[birth_index].copy(self.conditions)
        new_evol, new_adap = new_cell.update_cell(
            self.conditions,
            probability_to_evolve=evolution_probabilty,
            distance_mult=distance_mult,
            probability_to_adapt=adaptation_probability,
            probability_to_evolve_without_adapt=evolution_without_epigenesis,
        )
        
        self.cumulative_growth_rates[death_index:] = [self.cumulative_growth_rates[i] + new_cell.growth_rate - self.cells[death_index].growth_rate for i in range(death_index, self.n)]
            
        evol_dead_cell = self.cells[death_index].evolution
        evol_new_cell = new_cell.evolution

        self.sum_absolute_generation -= self.cells[death_index].absolute_generation
        self.sum_generation_since_last_evolution -= self.cells[death_index].generation_since_last_evolution
        self.total_growth_rate = self.cumulative_growth_rates[-1]
        if new_evol != -1:
            self.list_evolutions.append(new_evol)
            self.evolution_count[new_evol] = [(0,n_timesteps-1)]
            new_cell.evolution = new_evol 
            evol_new_cell = new_evol
            new_cell.generation_since_last_evolution = 0

        self.cells[death_index] = new_cell
        self.sum_absolute_generation += new_cell.absolute_generation
        self.sum_generation_since_last_evolution += new_cell.generation_since_last_evolution

        if new_adap != -1:
            self.n_adaptation += 1
        if evol_dead_cell == evol_new_cell:
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
    evolution_probabilty: float = 0.01,
    adaptation_probability: float = 0.01,
    evolution_without_epigenesis: float = 0.0001,
    distance_mult: float = 1.1,
):
    proportions_type = [sample.get_proportions_per_type()]
    growth_rate_by_type = [sample.get_mean_growth_rate_by_type()]
    growth_rates = [sample.get_mean_growth_rate_function()]
    absolute_generation = [sample.sum_absolute_generation/N]
    generation_since_last_evolution = [sample.sum_generation_since_last_evolution/N]
    current_time = 0
    timesteps = [0]
    n_timesteps = 1
    division_times = []
    for conditions, change_time in conditions_profile:
        sample.change_conditions(conditions)
        while current_time < change_time:
            next_time = compute_next_time_step(sample.n, sample.get_cumulative_growth_rate_function(), "lognormal") / sample.n
            current_time += next_time
            birth_rate = np.random.uniform(0, sample.get_cumulative_growth_rate_function())
            birth_index = np.searchsorted(sample.cumulative_growth_rates, birth_rate)
            division_times.append(next_time)
            death_index = np.random.randint(sample.n)
            sample.update(
                birth_index,
                death_index,
                n_timesteps,
                evolution_probabilty,
                adaptation_probability,
                evolution_without_epigenesis,
                distance_mult,

            )

            growth_rates.append(sample.get_mean_growth_rate_function())
            timesteps.append(current_time)
            absolute_generation.append(sample.sum_absolute_generation/N)
            generation_since_last_evolution.append(sample.sum_generation_since_last_evolution/N)
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
        sample.list_evolutions,
        absolute_generation,
        generation_since_last_evolution,
        division_times
    )


def main(
    first_evolution,
    numbers,
    conditions_profile: list[tuple[int, int]],
    probabilities: Optional[list[float]] = None,
    n_iter: int = 10,
):
    assert len(first_evolution) == len(numbers)
    for k in range(n_iter):
        np.random.seed(k)
        cells = []
        initial_conditions = conditions_profile[0][0]
        for i in range(len(first_evolution)):
            for j in range(numbers[i]):
                cells.append(
                    EvolutiveCells1D(i, first_evolution=first_evolution[i], conditions=initial_conditions)
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
                list_evolution,
                absolute_generation,
                generation_since_last_evolution,
                division_times  
            ) = Moran_process(
                sample,
                conditions_profile,
                adaptation_probability=probabilities[0],
                evolution_probabilty=probabilities[1],
                evolution_without_epigenesis=probabilities[2],
            )
        else:
            (
                timesteps,
                proportions_type,
                growth_rate_by_type,
                mean_growth_rates,
                list_evolution,
                absolute_generation,
                generation_since_last_evolution,
                division_times
            ) = Moran_process(sample, conditions_profile)

        print(f"Time elapsed: {time.time()-start} \n")

        top_10_evolutions = sample.find_top_evolutions()

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
        plt.hist(division_times, bins=150)
        plt.xlabel("Time")
        plt.ylabel("Number of divisions")
        plt.title("Division times")


        plt.show()
        (
                timesteps,
                proportions_type,
                proportion_evolution,
                growth_rate_by_type,
                mean_growth_rates,
                list_evolution,
                absolute_generation,
                generation_since_last_evolution
            ) = None, None, None, None, None, None, None, None
    return timesteps



#cProfile.run("main(first_evolution, numbers, conditions_profile, probabilities)", sort="tottime")

main(first_evolution, numbers, conditions_profile, probabilities)

