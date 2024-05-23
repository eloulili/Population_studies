# Import necessary libraries for typing annotations, mathematical operations, plotting, profiling, and timing.
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import cProfile
import time
import statistics

# Set the seed for NumPy's random number generator to ensure reproducibility.
np.random.seed(0)

# Define constants and simulation parameters.
N = 400
first_evolution = [[0,0,0,0,0,0]]  # Initial evolution state
adaptation_probability = 1  # Probability of adaptation
base_growth_rate = 0.5  # Base growth rate
stds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # Standard deviations for the evolution



n_iter = 5000  # Number of iterations for the simulation.



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



class EvolutiveCell:
    """Class to represent an evolutionary cell with specific characteristics and behaviors."""
    def __init__(self, type: int, epigenetic: list[float]):
        self.type = type
        self.epigenetic = epigenetic
        self.absolute_generation = 0
        self.previous_epigenetic = [self.epigenetic]

    # Methods for updating cell state, copying the cell, and converting cell information to a string are omitted for brevity.

# More classes and functions such as EvolutiveSample1D, Moran_process, and main are defined here.
# These functions handle simulation setup, execution, and data analysis, including adapting to new conditions, updating cell states, and visualizing results.

    def reproduce(self, probability_to_adapt: float = 0.1,
        stds = list[float]):

        assert len(stds) == len(self.epigenetic)
        new_cell = EvolutiveCell(self.type, epigenetic=self.epigenetic)
        new_cell.absolute_generation = self.absolute_generation +1
        new_cell.previous_epigenetic = self.previous_epigenetic.copy()
        if np.random.uniform() < probability_to_adapt:
            new_cell.previous_epigenetic.append([self.epigenetic])
            new_epigenetic = list((np.random.normal(self.epigenetic, stds)))
            new_cell.epigenetic = new_epigenetic
        return new_cell


class EvolutiveSample:

    def __init__(self, cells: list[EvolutiveCell], nb_types: int):
        self.cells = cells
        self.n = len(cells)
        self.cumulative_growth_rates = [] 
        self.list_evolutions = [cell.epigenetic for cell in self.cells if cell.epigenetic is not None]  

        # Variables used for tracking and analysis
        self.sum_absolute_generation = 0
        self.sum_evolution = cells[0].epigenetic.copy()
        for i in  range(1, len(cells)):
            for j in range(len(cells[i].epigenetic)):
                self.sum_evolution[j] += cells[i].epigenetic[j]
        
        self.genetic_tree = []


        self.nb_types = nb_types
        self.quantity_per_type = [None for _ in range(nb_types)]
        for type in range(nb_types):
            self.quantity_per_type[type] = sum([cell.type == type for cell in self.cells]) 


    def update(
        self,
        birth_index,
        stds = list[float],
        adaptation_probability: float = 0.1,
    ):
        
        # Update the sample by replacing a cell with a new one based on the birth and death indices, and possible evolution.
        # Update also tracking variables and statistics.
        new_cell = self.cells[birth_index].reproduce(adaptation_probability, stds)

        evol_new_cell = new_cell.epigenetic 
        
        self.genetic_tree.append(new_cell.previous_epigenetic + [new_cell.epigenetic])

        self.sum_absolute_generation += new_cell.absolute_generation


        if self.nb_types > 1:
            self.quantity_per_type[new_cell.type] += 1

        for i in range(len(evol_new_cell)):
            self.sum_evolution[i] += evol_new_cell[i]
        self.list_evolutions.append(evol_new_cell)
        self.cells.append(new_cell)
        self.n += 1

    def __str__(self):
        string = ""
        for cell in self.cells:
            string += str(cell) + "\n"

        return string
    
    
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
    sample: EvolutiveSample,
    adaptation_probability: float = 0.01,
    base_growth_rate: float = 0.5,
    stds = list[float],    
):
    """Simulate the Moran process with adaptation and loss of adaptation.
    To compute the next time step, we use an exponential distribution with the total growth rate as the rate parameter
    like in a gillespie algorithm."""

    quantity_type = [sample.quantity_per_type.copy()]
    absolute_generation = [sample.sum_absolute_generation/sample.n]
    mean_epigenetic = [sample.sum_evolution[i]/sample.n for i in range(len(sample.sum_evolution))]
    current_time = 0
    timesteps = [0]
    n_timesteps = 1
    while sample.n < N:
            next_time = np.random.exponential(1 / (sample.n * base_growth_rate))
            current_time += next_time
            birth_index = np.random.randint(sample.n)

            sample.update(
                birth_index = birth_index,
                adaptation_probability = adaptation_probability,
                stds = stds,
            )

            timesteps.append(current_time)
            absolute_generation.append(sample.sum_absolute_generation/sample.n)
            quantity_type.append(sample.quantity_per_type.copy())
            mean_epigenetic.append(sample.sum_evolution[i]/sample.n for i in range(len(sample.sum_evolution)))

            n_timesteps += 1  

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
        mean_epigenetic,
        absolute_generation,
    )


def main(
    first_evolution,
    adaptation_probability: float = 0.01,
    base_growth_rate: float = 0.5,
    stds = list[float],
    n_iter: int = 1,
):
    np.random.seed(0)
    cells = []
    cells = [EvolutiveCell( 0, epigenetic=first_evolution[i]) for i in range(len(first_evolution))]
    list_evolutions = []
    variance_deviation_per_simulation = []
    start = time.time()
    for k in range(n_iter):
        # Initialization, control the seed for reproducibility
        np.random.seed(k)
        cells = []
        cells = [EvolutiveCell( 0, epigenetic=first_evolution[i]) for i in range(len(first_evolution))]


        sample = EvolutiveSample(
            cells,
            len(first_evolution),
        )
        
        (timesteps,
        quantity_type,
        mean_epigenetic,
        absolute_generation,
            ) = Moran_process(
                sample,
                adaptation_probability= adaptation_probability,
                base_growth_rate= base_growth_rate,
                stds = stds,
            )
        list_evolutions.append(np.array(sample.list_evolutions).T)
        iteration_std = np.std(sample.list_evolutions, axis=0)
        variance_deviation_per_simulation.append(iteration_std)
        print(f"{100 * k/n_iter}%")




        # Plotting
        """""
        plt.figure()
        plt.plot(sample.list_evolutions)
        plt.legend()


        plt.figure()
        plt.plot(timesteps, absolute_generation, label="Mean absolute generation")
        plt.xlabel("Time")
        plt.ylabel("Mean absolute generation")
        plt.title("Mean absolute generation")


        plt.figure()
        plt.plot(timesteps, mean_epigenetic, label="Mean epigenetic")
        plt.xlabel("Time")
        plt.ylabel("Mean epigenetic")
        plt.title("Mean epigenetic")
        plt.legend()

        plt.figure()
        for i in range(len(total_genetic_tree)):
            plt.plot(range(len(total_genetic_tree[i][1])),total_genetic_tree[i][1] , label=f"Evolution {total_genetic_tree[i][0]}")
        plt.xlabel("Generation")
        plt.ylabel("Evolution")
        plt.title("Total epigenetic tree")

   

        plt.show()

        """
        (       timesteps,
                list_evolution,
                absolute_generation,
                mean_epigenetic,
                sample,
                total_genetic_tree,
            ) = None, None, None, None, None, None
    end = time.time()
    print(f"Time elapsed: {end - start}")
    print(f"Mean variance of deviation: {np.mean(variance_deviation_per_simulation, axis=0)}")
    print(f"Divided by individuals std: {np.mean(variance_deviation_per_simulation, axis=0)/stds}")
    return timesteps



#cProfile.run("main(first_evolution, numbers, conditions_profile, probabilities)", sort="tottime")

main(first_evolution, n_iter=n_iter, stds=stds, adaptation_probability=adaptation_probability, base_growth_rate=base_growth_rate)

