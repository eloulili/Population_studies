# Import necessary libraries for typing annotations, mathematical operations, plotting, profiling, and timing.
from typing import Optional
import numpy as np
import matplotlib.scale as scle
import matplotlib.pyplot as plt
import cProfile
import time

# Set the seed for NumPy's random number generator to ensure reproducibility.
np.random.seed(0)

"""
In this simulation, we run a Moran process with adaptation and loss of adaptation in a population of evolving cells.
Each cell has a type( family name), an epigenetic value, and a growth rate that depends on the epigenetic value and environmental conditions.
When a cell loses adaptation, it reverts to the last epigenetic value it had before adapting.
It is possible for a cell to fixate its epigenetic value, meaning it will never go back.
"""


# Define constants and simulation parameters.
N = 500
probabilities = [0.1, 0.3, 0.0005]  # probabilities for evolution and loss of adaptation, and fixation
first_evolution = [0]  # Initial evolution state
numbers = [500]  # Number of entities
conditions_profile = [(0,100)]  # Environmental conditions profile

# Constants for calculations within the growth rate functions.
DISTANT_COEFF = 1
CST_VALUE = 0.5
neutral_coefficient = np.exp(-CST_VALUE)  # Coefficient for adjusting growth based on a constant value.

smooth_coefficient = 600  # Coefficient for data smoothing operations.
std = 1 # Standard deviation for normal distribution used in simulations.

COND_COEF = 10  # Coefficient for condition-related calculations.
growth_rate_error = 0.0001  # Error term in growth rate calculations.
n_iter = 1  # Number of iterations for the simulation.

trigger = 0  # Trigger level for certain conditional checks.


def smooth_data(data, timesteps, smooth_coefficient: int):
    """Smooth data over a specified window of timesteps."""
    assert len(data) == len(timesteps)  # Ensure data and timesteps match in length.
    smoothed_data = [None] * (len(data) - smooth_coefficient)
    steps = [timesteps[i] - timesteps[i - 1] for i in range(1, len(data))]
    total_time = sum(steps[:smooth_coefficient])
    normalized_mean_data = sum([steps[i] * data[i] for i in range(smooth_coefficient)])
    for i in range(smooth_coefficient, len(data)):    
        smoothed_data[i-smooth_coefficient] = normalized_mean_data / total_time
        total_time += steps[i-1] - steps[i - smooth_coefficient -1]
        normalized_mean_data = normalized_mean_data - data[i - smooth_coefficient] * steps[i - smooth_coefficient-1] + data[i] * steps[i-1]
    return smoothed_data

def growth_rate_function(best_gene_distance, epigenetic, condition):
    """Calculate growth rate based on gene distance, epigenetic factors, and environmental conditions."""
    if epigenetic is None:
        return CST_VALUE
    return CST_VALUE + np.log(max(neutral_coefficient, 1 + DISTANT_COEFF * epigenetic)) / 20


def compute_next_time_step(n: int, total_growth_rate: float, type:str, std:float = 0.01) -> float:
    mean_time = 1/total_growth_rate
    local_std = std / np.sqrt(n)
    if type == "lognormal":
        sigma = np.sqrt(np.log(1 + local_std**2 / mean_time ** 2))
        mu = np.log(mean_time) - sigma ** 2 / 2  
        return np.random.lognormal(mean=mu , sigma=sigma)
    if type == "exponential":
        return np.random.exponential(mean_time)
        

class EvolutiveCells1D:
    """Class to represent an evolutionary cell with specific characteristics and behaviors."""
    def __init__(self, type: int, epigenetic: Optional[float] = None, conditions=0, growth_rate_error=0.,  growth_rate: Optional[float] = None):
        self.type = type
        self.epigenetic = epigenetic
        if growth_rate is None and epigenetic is not None:
            self.growth_rate = growth_rate_function(abs(epigenetic - conditions), self.epigenetic, conditions) + growth_rate_error
        elif growth_rate is None:
            self.growth_rate = CST_VALUE + growth_rate_error        
        else:
            self.growth_rate = growth_rate + growth_rate_error
        self.gr_error = growth_rate_error
        self.epigenetic_generation = 0
        self.absolute_generation = 0
        self.generation_since_last_mutation = 0
        self.previous_epigenetic = [self.epigenetic]
        self.base_epigenetic = 0

    # Methods for updating cell state, copying the cell, and converting cell information to a string are omitted for brevity.

# More classes and functions such as EvolutiveSample1D, Moran_process, and main are defined here.
# These functions handle simulation setup, execution, and data analysis, including adapting to new conditions, updating cell states, and visualizing results.

    def update_cell(
        self,
        conditions,
        probability_to_adapt: float = 0.1,
        adaptation_loss_probability: float = 0.5,
        fix_probability: float = 0.0001,
        std: float = 0.1, # Standard deviation of an evolution step
        quick_growth_rates: dict = {},
        max_type: int = 1
    ) -> tuple[bool, bool]:
        """Update cell based on environmental conditions and probabilities of adaptation and adaptation loss."""
        
        new_epigenetic = None
        has_lost_adaptation = False
        has_fixed = False
        if self.epigenetic != self.base_epigenetic:
            if self.epigenetic != self.base_epigenetic and np.random.uniform(0, 1) < fix_probability:
                self.base_epigenetic = self.epigenetic + 0.
                quick_growth_rates[self.base_epigenetic] = growth_rate_function(abs(self.base_epigenetic - conditions), self.base_epigenetic, conditions)
                self.previous_epigenetic = [self.base_epigenetic]
                self.epigenetic_generation = 0
                max_type += 1
                self.type = max_type
                has_fixed = True
            
        if self.epigenetic != self.base_epigenetic:
            if np.random.uniform(0, 1) < adaptation_loss_probability:
            # If the cell loses adaptation, update the epigenetic value and generation count.
                has_lost_adaptation = True
                self.epigenetic_generation -= 1
                if self.previous_epigenetic :
                    # If the cell has previous epigenetic values, revert to the last one.
                    self.epigenetic = self.previous_epigenetic.pop(0)
                else:
                    self.epigenetic = self.base_epigenetic

            

        if np.random.uniform(0, 1) < probability_to_adapt:
            # If the cell adapts, update the epigenetic value and generation count.:
                if self.epigenetic != self.base_epigenetic:
                    self.previous_epigenetic.append(self.epigenetic)

                new_epigenetic = np.random.normal(self.epigenetic, std)  # Feel free to change the distribution
                
                self.epigenetic = new_epigenetic
                self.epigenetic_generation += 1
        best_distance = None
        if self.epigenetic == self.base_epigenetic :
            # If the cell has no epigenetic value, set the growth rate to a constant value with an error term.
            self.generation_since_last_mutation = 0
            self.growth_rate = quick_growth_rates[self.base_epigenetic] + self.gr_error

        elif has_lost_adaptation or new_epigenetic is not None:
            # If the cell has lost adaptation or adapted, update the growth rate based on the new epigenetic value.
            best_distance = abs(self.epigenetic - conditions)
            self.growth_rate = growth_rate_function(best_distance, self.epigenetic, conditions) + self.gr_error

        # If the cell did not adapt or lose adaptation, the growth rate has already been set in the initialization.      

        return new_epigenetic is not None, self.epigenetic != self.base_epigenetic, has_fixed

    def copy(self, conditions):
        additive_error = np.random.normal(0, growth_rate_error)
        new_cell = EvolutiveCells1D(self.type, conditions=conditions, epigenetic=self.epigenetic, growth_rate=self.growth_rate - self.gr_error, growth_rate_error=additive_error + self.gr_error)
        new_cell.absolute_generation = self.absolute_generation +1
        new_cell.generation_since_last_mutation = self.generation_since_last_mutation +1
        new_cell.epigenetic_generation = self.epigenetic_generation
        new_cell.previous_epigenetic = self.previous_epigenetic.copy()
        new_cell.base_epigenetic = self.base_epigenetic
        return new_cell


class EvolutiveSample1D:

    def __init__(self, cells: list[EvolutiveCells1D], nb_types: int, conditions=0):
        self.cells = cells
        self.n = len(cells)
        self.cumulative_growth_rates = [] 
        self.conditions = conditions
        self.evolution_count  : dict[float,tuple[int,int]] = dict()
        for cell in self.cells:
            if cell.epigenetic not in self.evolution_count:
                if cell.epigenetic is not None:
                    self.evolution_count[cell.epigenetic] = [(1,0)]
            else:
                self.evolution_count[cell.epigenetic][0] = (self.evolution_count[cell.epigenetic][0][0] +1 ,0)

        cumul = 0.0
        for i in range(self.n):
            cumul += self.cells[i].growth_rate
            self.cumulative_growth_rates.append(cumul)

        # Variables used for tracking and analysis
        self.total_growth_rate = cumul
        self.sum_epigenetic_generation = 0
        self.sum_absolute_generation = 0
        self.sum_generation_since_last_evolution = 0
        self.sum_evolution = sum([cell.epigenetic for cell in self.cells if cell.epigenetic is not None])

        self.genetic_tree = []

        self.list_evolutions = list(set([cell.epigenetic for cell in self.cells]))
        if None in self.list_evolutions:
            self.list_evolutions.remove(None)
        self.quantity_with_epigenetic = [sum([cell.epigenetic != cell.base_epigenetic for cell in self.cells])]

        self.max_evolution = [max([-1] + [cell.epigenetic for cell in self.cells if cell.epigenetic is not None])]

        self.nb_types = nb_types
        self.quantity_per_type = [None for _ in range(nb_types)]
        for type in range(nb_types):
            self.quantity_per_type[type] = sum([cell.type == type for cell in self.cells]) 

        self.growth_rate_per_type = [None for _ in range(nb_types)]
        for type in range(nb_types):
            self.growth_rate_per_type[type] = sum([cell.growth_rate for cell in self.cells if cell.type == type]) / sum([cell.type == type for cell in self.cells])

        self.quick_growth_rate = dict()
        for cell in self.cells:
                self.quick_growth_rate[cell.base_epigenetic] = growth_rate_function(abs(cell.base_epigenetic - self.conditions), cell.base_epigenetic, self.conditions)

    def change_conditions(self, conditions):
        "Change environmental conditions and update growth rates accordingly."
        self.conditions = conditions
        for cell in self.cells:
            best_distance = None
            if cell.epigenetic is not None:

                best_distance = abs(cell.epigenetic - conditions)
            cell.growth_rate = growth_rate_function(
                best_distance, cell.epigenetic, conditions
            ) + cell.gr_error
        cumul = 0.0
        for i in range(len(self.cells)):
            cumul += self.cells[i].growth_rate
            self.cumulative_growth_rates[i]
        self.total_growth_rate = cumul

        for base_epigenetic in self.quick_growth_rate.keys():
            self.quick_growth_rate[base_epigenetic] = growth_rate_function(abs(base_epigenetic - self.conditions), base_epigenetic, self.conditions)


    #TODO : The two following functions can be optimized by using a dictionary to store the growth rate by type
    # instead of computing it each time
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


    def get_quantity_per_evolution(self, evolution_list):
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
        fixation_probability: float = 0.0001,
    ):
        
        # Update the sample by replacing a cell with a new one based on the birth and death indices, and possible evolution.
        # Update also tracking variables and statistics.
        new_cell = self.cells[birth_index].copy(self.conditions)
        has_new_epigenetic, has_epigenetic, has_fixed = new_cell.update_cell(
            self.conditions,
            probability_to_adapt=adaptation_probability,
            adaptation_loss_probability=adaptation_loss_probability,
            fix_probability=fixation_probability,
            std=std,
            quick_growth_rates=self.quick_growth_rate,
            max_type=self.nb_types
        )
        self.max_evolution.append(self.max_evolution[-1])
        self.cumulative_growth_rates[death_index:] = [self.cumulative_growth_rates[i] + new_cell.growth_rate - self.cells[death_index].growth_rate for i in range(death_index, self.n)]
        dead_cell = self.cells[death_index]
        
        if dead_cell.epigenetic == self.max_evolution[-2] and sum([cell.epigenetic == self.cells[death_index].epigenetic for cell in self.cells]) == 1:
            self.max_evolution[-1] = max([-1] + [cell.epigenetic for cell in self.cells if cell.epigenetic is not None and cell.epigenetic != self.cells[death_index].epigenetic])
        if dead_cell.epigenetic != dead_cell.base_epigenetic:   
            self.quantity_with_epigenetic.append(self.quantity_with_epigenetic[-1] - 1)
        else:
            self.quantity_with_epigenetic.append(self.quantity_with_epigenetic[-1])


        if has_epigenetic:
            self.quantity_with_epigenetic[-1] += 1
        else:
            new_cell.generation_since_last_mutation = 0
        evol_new_cell = new_cell.epigenetic 

        
        evol_dead_cell = self.cells[death_index].epigenetic


        self.sum_absolute_generation -= self.cells[death_index].absolute_generation
        self.sum_generation_since_last_evolution -= self.cells[death_index].generation_since_last_mutation
        self.sum_epigenetic_generation -= self.cells[death_index].epigenetic_generation
        self.total_growth_rate = self.cumulative_growth_rates[-1]
        if has_new_epigenetic:
            self.list_evolutions.append(evol_new_cell)
            self.evolution_count[evol_new_cell] = [(0,n_timesteps-1)]
            new_cell.generation_since_last_mutation = 0
            self.genetic_tree.append(new_cell.previous_epigenetic + [new_cell.epigenetic])
            if evol_new_cell > self.max_evolution[-2]:
                self.max_evolution[-1] = evol_new_cell

        self.cells[death_index] = new_cell
        self.sum_absolute_generation += new_cell.absolute_generation
        self.sum_generation_since_last_evolution += new_cell.generation_since_last_mutation
        self.sum_epigenetic_generation += new_cell.epigenetic_generation

        if has_fixed:
                self.sum_epigenetic_generation = sum([cell.epigenetic_generation for cell in self.cells])
                self.quantity_per_type.append(0)
                self.growth_rate_per_type.append(0)
                print("Fixed")

        if self.nb_types > 1:
            self.quantity_per_type[dead_cell.type] -= 1
            self.quantity_per_type[new_cell.type] += 1
            self.growth_rate_per_type[new_cell.type] = ((self.quantity_per_type[new_cell.type] -1)* self.growth_rate_per_type[new_cell.type] +  new_cell.growth_rate)/ self.quantity_per_type[new_cell.type]
            if self.quantity_per_type[dead_cell.type] != 0:
                    self.growth_rate_per_type[dead_cell.type] = ((self.quantity_per_type[dead_cell.type]  +1)* self.growth_rate_per_type[dead_cell.type]  - dead_cell.growth_rate)/ self.quantity_per_type[dead_cell.type]
            else:
                    self.growth_rate_per_type[dead_cell.type] = 0
                    self.nb_types -= 1

        if evol_dead_cell != evol_new_cell :
            self.evolution_count[evol_dead_cell].append((self.evolution_count[evol_dead_cell][-1][0]-1,n_timesteps))
            self.evolution_count[evol_new_cell].append((self.evolution_count[evol_new_cell][-1][0]+1,n_timesteps))
        self.sum_evolution += evol_new_cell - evol_dead_cell


        


    def __str__(self):
        string = ""
        for cell in self.cells:
            string += str(cell) + "\n"

        return string
    
    def find_top_evolutions(self, n_top=10):
        """Find the top n_top evolutions based on the highest quantity reached for each evolution."""
        # Initialisation
        best_quantity_per_evolution = {}

        for rank,(evolution, highest_point) in enumerate(self.evolution_count.items()):
            # Find the highest quantity reached for each evolution
            max_quantity = max(highest_point, key=lambda x: x[0])[0]
            best_quantity_per_evolution[evolution] = max_quantity , rank

        # Sort the evolutions by the highest quantity reached
        sorted_evolutions = sorted(best_quantity_per_evolution.items(), key=lambda item: item[1][0], reverse=True)

        # Get the top n_top evolutions
        top_evolutions = [evolution for evolution, quantity in sorted_evolutions[:n_top]]

        return top_evolutions
    
    def get_ascendance(self, list_evolution):
        """Get the ascendance of a list of evolutions."""
        tree = []
        for evolution in list_evolution:
            for tree_evolution in self.genetic_tree:
                if evolution == tree_evolution[-1]:
                    tree.append((evolution, tree_evolution))
                    break
        return tree



def Moran_process(
    sample: EvolutiveSample1D,
    conditions_profile: list[tuple[int, int]],
    adaptation_probability: float = 0.01,
    adaptation_loss_probability: float = 0.3,
    fixation_probability: float = 0.0001,
):
    """Simulate the Moran process with adaptation and loss of adaptation.
    To compute the next time step, we use an exponential distribution with the total growth rate as the rate parameter
    like in a gillespie algorithm."""

    quantity_type = [sample.quantity_per_type.copy()]
    growth_rate_by_type = [sample.get_mean_growth_rate_by_type()]
    growth_rates = [sample.total_growth_rate/N]
    absolute_generation = [sample.sum_absolute_generation/N]
    generation_since_last_evolution = [sample.sum_generation_since_last_evolution/N]
    mean_epigenetic_generation = [sample.sum_epigenetic_generation/N]
    mean_epigenetic = [sample.sum_evolution/N]
    current_time = 0
    timesteps = [0]
    n_timesteps = 1
    division_times = []
    for conditions, change_time in conditions_profile:
        sample.change_conditions(conditions)
        while current_time < change_time:
            next_time = compute_next_time_step(N, sample.total_growth_rate, "lognormal")
            current_time += next_time
            birth_rate = np.random.uniform(0, sample.total_growth_rate)
            birth_index = np.searchsorted(sample.cumulative_growth_rates, birth_rate) 
            if next_time < 2 *  1 / CST_VALUE:
                division_times.append(next_time*N )

            death_index = np.random.randint(sample.n)
            sample.update(
                birth_index,
                death_index,
                n_timesteps,
                adaptation_probability,
                adaptation_loss_probability,
                fixation_probability,
            )

            growth_rates.append(sample.total_growth_rate/N)
            timesteps.append(current_time)
            absolute_generation.append(sample.sum_absolute_generation/N)
            generation_since_last_evolution.append(sample.sum_generation_since_last_evolution/N)
            mean_epigenetic_generation.append(sample.sum_epigenetic_generation/N)
            quantity_type.append(sample.quantity_per_type.copy())
            #growth_rate_by_type.append(sample.growth_rate_per_type.copy())
            mean_epigenetic.append(sample.sum_evolution/N)

            n_timesteps += 1
    analyzed_epigenetic = []
    last_evolution_per_tree = [genetic_tree[-1] for genetic_tree in sample.genetic_tree]
    for cell in sample.cells:  
        if cell.epigenetic is not None and cell.epigenetic not in analyzed_epigenetic and cell.epigenetic not in last_evolution_per_tree:
            analyzed_epigenetic.append(cell.epigenetic)
            sample.genetic_tree.append(cell.previous_epigenetic + [cell.epigenetic])    

    transpose_quantity_type = [[] * sample.nb_types] 
    """""
    for i in range(len(quantity_type)):
        for j in range(sample.nb_types):
            if len(transpose_quantity_type) <= j:
                break
            transpose_quantity_type[j].append(quantity_type[i][j])
    """


    return (
        timesteps,
        transpose_quantity_type,
        growth_rate_by_type,
        growth_rates,
        mean_epigenetic,
        sample.list_evolutions,
        absolute_generation,
        generation_since_last_evolution,
        mean_epigenetic_generation,
        division_times
    )


def main(
    first_evolution,
    numbers,
    conditions_profile: list[tuple[int, int]],
    probabilities: Optional[list[float]] = None,
    n_iter: int = 1,
):
    assert len(first_evolution) == len(numbers)
    np.random.seed(0)
    cells = []
    initial_conditions = conditions_profile[0][0]
    for i in range(len(first_evolution)):
            for j in range(numbers[i]):
                cells.append(
                    EvolutiveCells1D(type=i, conditions=initial_conditions)
                )
    
    trigger_time = []


    for k in range(n_iter):
        # Initialization, control the seed for reproducibility
        np.random.seed(k)
        cells = []
        initial_conditions = conditions_profile[0][0]
        for i in range(len(first_evolution)):
            for j in range(numbers[i]):
                cells.append(
                    EvolutiveCells1D(i, conditions=initial_conditions, epigenetic=first_evolution[i])
                )

        sample = EvolutiveSample1D(
            cells,
            len(first_evolution),
            conditions=initial_conditions
        )
        
        start = time.time()
        if probabilities != None:
            (
                timesteps,
        quantity_type,
        growth_rate_by_type,
        mean_growth_rates,
        mean_epigenetic,
        sample.list_evolutions,
        absolute_generation,
        generation_since_last_evolution,
        mean_epigenetic_generation,
        division_times
            ) = Moran_process(
                sample,
                conditions_profile,
                adaptation_probability=probabilities[0],
                adaptation_loss_probability=probabilities[1],
                fixation_probability=probabilities[2],
            )
        else:
            (
                timesteps,
        quantity_type,
        growth_rate_by_type,
        mean_growth_rates,
        mean_epigenetic,
        sample.list_evolutions,
        absolute_generation,
        generation_since_last_evolution,
        mean_epigenetic_generation
            ) = Moran_process(sample, conditions_profile)

        print(f"Time elapsed: {time.time()-start} \n")
        start = time.time()
        # Find the top 10 evolutions based on the highest quantity reached for each evolution
        # because it is impossible to plot all the evolutions
        top_10_evolutions = sample.find_top_evolutions( n_top=10)
        genetic_tree = sample.get_ascendance(top_10_evolutions)

        top_100_evolutions = sample.find_top_evolutions( n_top=100)
        total_genetic_tree = sample.get_ascendance(top_100_evolutions)

        # Smoothing data for better visualization     
        smoothed_growth_rates = smooth_data(mean_growth_rates, timesteps, smooth_coefficient=smooth_coefficient)
        smoothed_max_evolution = smooth_data(sample.max_evolution, timesteps, smooth_coefficient=smooth_coefficient)
        smoothed_quantity_with_epigenetics = smooth_data(sample.quantity_with_epigenetic, timesteps, smooth_coefficient=smooth_coefficient)
        smoothed_mean_epigenetic = smooth_data(mean_epigenetic, timesteps, smooth_coefficient=smooth_coefficient)
        smoothed_mean_generation_since_last_evolution = smooth_data(generation_since_last_evolution, timesteps, smooth_coefficient=smooth_coefficient)
        
        trigger_index = np.searchsorted(smoothed_mean_generation_since_last_evolution,  trigger)
        trigger_time.append(timesteps[smooth_coefficient//2:-smooth_coefficient//2][trigger_index])


        print(f"Post process time: {time.time()-start} \n")

        # Plotting
        
        plt.figure()
        for evolution in top_10_evolutions:
            quantity_evolution =  []
            first_appearance = sample.evolution_count[evolution][0][1]
            last_appearance = sample.evolution_count[evolution][-1][1]
            previous_change=first_appearance
            for n, t_change in sample.evolution_count[evolution]:
                quantity_evolution.extend([n]*(t_change-previous_change))
                previous_change = t_change
            plt.plot(timesteps[first_appearance:last_appearance], quantity_evolution ,label=f"Evolution {evolution}")
        plt.xlabel("Time")
        plt.ylabel("Proportion of cells")
        plt.title("Proportion of cells per evolution")
        plt.legend()

        plt.figure()
        plt.plot(timesteps, mean_growth_rates, label="Mean growth rate")
        plt.plot(timesteps[smooth_coefficient//2:-smooth_coefficient//2], smoothed_growth_rates, label="Smoothed mean growth rate")
        #plt.plot(timesteps[smooth_coefficient//2:-smooth_coefficient//2], epigenetic_based_growth_rate, label="Epigenetic based growth rate")
        plt.xlabel("Time")
        plt.ylabel("Mean growth rate")
        plt.title("Mean growth rate")
        plt.legend()

        plt.figure()
        plt.plot(timesteps, absolute_generation, label="Mean absolute generation")
        plt.xlabel("Time")
        plt.ylabel("Mean absolute generation")
        plt.title("Mean absolute generation")

        plt.figure()
        plt.plot(timesteps, generation_since_last_evolution, label="Mean generation since last evolution")
        plt.plot(timesteps[smooth_coefficient//2:-smooth_coefficient//2], smoothed_mean_generation_since_last_evolution, label="Smoothed mean absolute generation")
        plt.plot(timesteps[smooth_coefficient//2:-smooth_coefficient//2], [trigger]*len(timesteps[smooth_coefficient//2:-smooth_coefficient//2]), linestyle="--", color="red")
        plt.xlabel("Time")
        plt.ylabel("Mean generation since last evolution")
        plt.title("Mean generation since last evolution")
        plt.legend()

        plt.figure()
        plt.plot(timesteps, mean_epigenetic, label="Mean epigenetic")
        plt.plot(timesteps[smooth_coefficient//2:-smooth_coefficient//2], smoothed_mean_epigenetic, label="Smoothed mean epigenetic")
        plt.xlabel("Time")
        plt.ylabel("Mean epigenetic")
        plt.title("Mean epigenetic")
        plt.legend()

        plt.figure()
        plt.plot(timesteps, sample.quantity_with_epigenetic, label="Proportion with epigenetic")
        plt.plot(timesteps[smooth_coefficient//2:-smooth_coefficient//2], smoothed_quantity_with_epigenetics, label="Smoothed quantity with epigenetic")
        plt.xlabel("Time")
        plt.ylabel("Proportion with epigenetic")
        plt.title("Proportion with epigenetic")
        plt.legend()

        plt.figure()
        plt.plot(timesteps, sample.max_evolution, label="Max evolution")
        plt.plot(timesteps[smooth_coefficient//2:-smooth_coefficient//2], smoothed_max_evolution, label="Smoothed max evolution")
        plt.xlabel("Time")
        plt.ylabel("Max evolution")
        plt.title("Max evolution")
        plt.legend()

        plt.figure()
        plt.plot(timesteps, mean_epigenetic_generation, label="Mean epigenetic generation")
        plt.xlabel("Time")
        plt.ylabel("Mean epigenetic generation")
        plt.title("Mean epigenetic generation")

        plt.figure()
        plt.plot(absolute_generation, mean_growth_rates)
        plt.xlabel("Absolute generation")
        plt.ylabel("Mean growth rate")
        plt.title("Mean growth rate by absolute generation")

        plt.figure()
        for i in range(len(genetic_tree)):
            plt.plot(range(len(genetic_tree[i][1])),genetic_tree[i][1] , label=f"Evolution {genetic_tree[i][0]}")
        plt.xlabel("Generation")
        plt.ylabel("Evolution")
        plt.title("Epigenetic tree")
        plt.legend()

        plt.figure()
        for i in range(len(total_genetic_tree)):
            plt.plot(range(len(total_genetic_tree[i][1])),total_genetic_tree[i][1] , label=f"Evolution {total_genetic_tree[i][0]}")
        plt.xlabel("Generation")
        plt.ylabel("Evolution")
        plt.title("Total epigenetic tree")

        plt.figure()
        plt.hist(division_times, bins=50)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.title("Division time distribution")

   

        plt.show()
        (       timesteps,
                quantity_evolution,
                growth_rate_by_type,
                mean_growth_rates,
                list_evolution,
                absolute_generation,
                generation_since_last_evolution,
                mean_epigenetic,
                sample,
                smoothed_growth_rates,
                smoothed_max_evolution,
                smoothed_quantity_with_epigenetics,
                smoothed_mean_epigenetic,
                smoothed_mean_generation_since_last_evolution,
                total_genetic_tree,
                genetic_tree,
                division_times
            ) = None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

    print(f"First increasing tendency: {trigger_time}")
    print(f"Mean first increasing tendency: {sum(trigger_time)/n_iter}")
    return timesteps



#cProfile.run("main(first_evolution, numbers, conditions_profile, probabilities)", sort="tottime")

main(first_evolution, numbers, conditions_profile, probabilities, n_iter=n_iter)

