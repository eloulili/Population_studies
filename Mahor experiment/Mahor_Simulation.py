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
N = 40000
first_evolution = [[0,0,0,0,0,0]]  # Initial evolution state
adaptation_probability = 1  # Probability of adaptation
base_growth_rate = 0.5  # Base growth rate
stds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # Standard deviations for the evolution
n_iter = 1  # Number of iterations


MULTIPLICATIVE = False
SQUARE_ROOT = False

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
        stds = list[float],
        square_root: bool = SQUARE_ROOT,
        use_multiplicative: bool = MULTIPLICATIVE,
):

        assert len(stds) == len(self.epigenetic)
        new_cell = EvolutiveCell(self.type, epigenetic=self.epigenetic)
        new_cell.absolute_generation = self.absolute_generation +1
        new_cell.previous_epigenetic = self.previous_epigenetic.copy()
        if np.random.uniform() < probability_to_adapt:
            new_epigenetic =  self.epigenetic.copy()
            if not use_multiplicative:
                new_cell.previous_epigenetic.append([self.epigenetic])
                if square_root:
                    new_epigenetic =  self.epigenetic.copy()

                    addition = []
                    for std in stds:
                        a = np.random.normal(0, std) 
                        addition.append(a**2 * np.sign(a))
                    for i in range(len(new_epigenetic)):
                        new_epigenetic[i] += addition[i]
                else:
                    new_epigenetic = list(np.random.normal(self.epigenetic, stds))
            else:
                
                for i in range(len(self.epigenetic)):
                    multiplicative = 1
                    a = np.random.normal(0, stds[i])
                    if square_root:
                        multiplicative += a**2 * np.sign(a)
                    else:
                        multiplicative += a
                    new_epigenetic[i] = self.epigenetic[i] * multiplicative
                    
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
    max_population_size: int = N
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
    while sample.n < max_population_size:
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
    max_population_size: int = N
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
                max_population_size = max_population_size,
            )
        list_evolutions.append(np.array(sample.list_evolutions).T)
        iteration_std = np.var(sample.list_evolutions, axis=0)
        variance_deviation_per_simulation.append(iteration_std)



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

    list_deviation_replicates = []
    evolution_first_std = []
    for i in range(len(stds)):
        evolutions_through_replicates = []
        for j in range(n_iter):
            for k in range(max_population_size):
                evol = list_evolutions[j][i][k]
                if evol != 0:
                    evolutions_through_replicates.append(list_evolutions[j][i][k])
        list_deviation_replicates.append(np.var(evolutions_through_replicates))
        if i == 0:
            evolution_first_std = evolutions_through_replicates

    vars =[]
    for std in stds:
        vars.append(std**2)
    print(f"Time elapsed: {end - start}")
    print(f"Mean variance of deviation: {np.mean(variance_deviation_per_simulation, axis=0)}")
    print(f"Mean deviation through replicates: {list_deviation_replicates}")
    print(f"Ratio of mean deviation through replicates with individual variance: {np.mean(variance_deviation_per_simulation, axis=0)/list_deviation_replicates}")
    print(f"Mean ratio of mean deviation through replicates: {np.mean(variance_deviation_per_simulation, axis=0)/vars}")
    print(f"Mean ratio of mean deviation through replicates: {np.mean(np.mean(variance_deviation_per_simulation, axis=0)/vars)}")
    print(f"Ratio of mean deviation through replicates: {np.mean(variance_deviation_per_simulation, axis=0)/list_deviation_replicates}")
    print(f"Mean ratio of mean deviation through replicates: {np.mean(np.mean(variance_deviation_per_simulation, axis=0)/list_deviation_replicates)}")
    print(f"Parameters: stds =  {stds}, adaptation_probability = {adaptation_probability}, growth_rate = {base_growth_rate}, \n")
    return np.mean(list_deviation_replicates/np.mean(variance_deviation_per_simulation, axis=0)),np.mean(vars/np.mean(variance_deviation_per_simulation, axis=0)) ,evolution_first_std




def main_lior(
    adaptation_probability: float = 0.01,
    base_growth_rate: float = 0.5,
    stds: float = [0.01],
    n_1: int = 1,
    dillution_1: int = 1e3,
    n_2: int = 1,
    dillution_2: int = 1e4,
    seed: int = 0,
    verbose: bool = True) :

    np.random.seed(seed)
    list_evolutions = []
    variance_deviation_per_simulation = []
    std_per_simulation = []
    mean_per_simulation = []
    start = time.time()

    cells = [EvolutiveCell( 0, [1])]


    otiginal_sample = EvolutiveSample(
            cells,
            len(first_evolution),
        )
    max_population_size = 2**n_1  
    (timesteps,
        quantity_type,
        mean_epigenetic,
        absolute_generation,
            ) = Moran_process(
                otiginal_sample,
                adaptation_probability= adaptation_probability,
                base_growth_rate= base_growth_rate,
                stds = stds,
                max_population_size = max_population_size,
            )
    
    max_population_size = 2**n_2
    trigger = 0.05
    start = time.time()
    for i in range(int(dillution_1)):
        random_index = np.random.randint(otiginal_sample.n)
        original_cell = otiginal_sample.cells[random_index]
        new_sample = EvolutiveSample(cells = [original_cell], nb_types = 1)
        (timesteps, 
        quantity_type,
        mean_epigenetic, 
        absolute_generation) = Moran_process(new_sample, adaptation_probability= adaptation_probability, base_growth_rate= base_growth_rate, stds = stds, max_population_size = max_population_size)
         

        evolutions = (np.array(new_sample.list_evolutions).T)
        random_indexes = np.random.randint(new_sample.n, size = int(dillution_2))
        list_evolutions.append(evolutions[0][random_indexes])
        mean_per_simulation.append(np.mean(evolutions[0]))
        if i / dillution_1 > trigger and verbose:
            print(f"Progress: {trigger * 100} %, Time elapsed: {time.time() - start}")
            trigger += 0.05
    total_populations = []
    for k in range(len(list_evolutions)):
        variance_deviation_per_simulation.append(np.var(list_evolutions[k]))
        for j in range(len(list_evolutions[k])):
            total_populations.append(list_evolutions[k][j])
    total_variance = np.var(total_populations)
    variance_inv = 1/np.array(variance_deviation_per_simulation)
    var_M = stds[0]**2 * adaptation_probability

    end = time.time()
    
    print(f"Mean Variance per simulation: {np.mean(variance_deviation_per_simulation, axis=0)}")
    print(f"Time elapsed: {end - start}")
    print(f"Variance total population: {total_variance}\n")
    #print(f"Mean variance of deviation : {np.mean(variance_deviation_per_simulation, axis=0)}")
    #print(f" Expected ratio : { 1 + n_1/(n_2 - 1) * (1 + (np.mean(variance_deviation_per_simulation, axis=0) / mean_per_simulation)**2 ) }")
    
    CV_2 = (np.std(variance_deviation_per_simulation) / np.mean(variance_deviation_per_simulation))**2
    expected_ratio = 1 + n_1/(n_2 - 1) * (1 + CV_2)
    expected_ratio_v2 = 1 + var_M * (n_2 - 1) * np.mean(variance_inv)
    effective_ratio = total_variance / np.mean(variance_deviation_per_simulation, axis=0)
    std_ratio = np.sqrt(CV_2)*(n_2 -1)/n_1
    
    print(f" CV² :{CV_2}")
    print(f"Mean_expected_ratio : {expected_ratio }")
    print(f"Ecpected ratio v2 : {expected_ratio_v2 }")
    print(f"STD on Ratio : {std_ratio}")
    #print(f"Ratio of mean deviation through replicates: {np.mean(variance_deviation_per_simulation, axis=0)/total_variance}")
    print(f"Effective Ratio: {effective_ratio}")
    print(f"Is in the range: {expected_ratio - 1.96*std_ratio<= effective_ratio  and effective_ratio <= expected_ratio + 1.96*std_ratio}")
    
    print(f"Parameters: stds =  {stds}, adaptation_probability = {adaptation_probability}, growth_rate = {base_growth_rate}, n_1 = {n_1}, dillution_1 = {dillution_1}, n_2 = {n_2}, dillution_2 = {dillution_2} ")

    return expected_ratio_v2 - 1.96*std_ratio<= effective_ratio  and effective_ratio <= expected_ratio_v2 + 1.96*std_ratio,expected_ratio - 1.96*std_ratio<= effective_ratio  and effective_ratio <= expected_ratio + 1.96*std_ratio, effective_ratio, expected_ratio, expected_ratio_v2, std_ratio

n_in_range = 0
n_in_range_v2 = 0
for seed in range(100):
    is_in_range_v2,is_in_range, effective_ratio, expected_ratio, expected_ratio_v2, std = main_lior(adaptation_probability= 1., base_growth_rate=base_growth_rate, stds = [0.1], n_1=15, dillution_1= 2000, n_2=8, dillution_2=256, seed=seed) 
    if is_in_range:
        n_in_range += 1

    if is_in_range_v2:
        n_in_range_v2 += 1
print(f"Number of simulations in range: {n_in_range}")
print(f"Percentage of simulations in range: {n_in_range/100}")
print(f"Number of simulations in range v2: {n_in_range_v2}")
print(f"Percentage of simulations in range v2: {n_in_range_v2/100}")

"""
for n_1 in [15, 16, 17, 18, 19, 20, 21]:
    n_in_range = 0
    n_in_range_v2 = 0
    effective_ratios = []
    expected_ratios = []
    expected_ratios_v2 = []
    std_ratios = []
    for seed in range(10):
        is_in_range_v2,is_in_range, effective_ratio, expected_ratio, expected_ratio_v2, std = main_lior(adaptation_probability= adaptation_probability, base_growth_rate=base_growth_rate, stds = [0.1], n_1=n_1, dillution_1=500, n_2=12, dillution_2=2048, seed=seed, verbose=False)
        effective_ratios.append(effective_ratio)
        expected_ratios.append(expected_ratio)
        expected_ratios_v2.append(expected_ratio_v2)
        std_ratios.append(std)
        if is_in_range:
            n_in_range += 1
        if is_in_range_v2:
            n_in_range_v2 += 1
    print(f"Number of simulations in range: {n_in_range}")
    print(f"Percentage of simulations in range: {n_in_range/10}")
    print(f"Number of simulations in range v2: {n_in_range_v2}")
    print(f"Percentage of simulations in range v2: {n_in_range_v2/10}")
    #print(f"Expected ratio: {expected_ratios}")
    #print(f"Expected ratio v2: {expected_ratios_v2}")
    #print(f"Effective ratio: {effective_ratios}")
    print(f"STD on ratio: {std_ratios}")
    print(f"Mean effective ratio: {np.mean(effective_ratios)}")
    print(f"Mean expected ratio: {np.mean(expected_ratios)}")
    print(f"Mean expected ratio v2: {np.mean(expected_ratios_v2)}")
    print(f"Mean std ratio: {np.mean(std_ratios)}")
"""



"""""

ratio, ratio_individual, evolutions= main(first_evolution, n_iter=n_iter, stds=stds, adaptation_probability=adaptation_probability, base_growth_rate=base_growth_rate, max_population_size=N)
print(ratio)
plt.hist(evolutions, bins=100)
plt.show()


ratios_per_probability = [] 
individual_ratios_per_probability = []
for adaptation_probability in [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]:
    ratios_per_probability.append(main(first_evolution, n_iter=n_iter, stds=stds, adaptation_probability=adaptation_probability, base_growth_rate=base_growth_rate)[0])
    individual_ratios_per_probability.append(main(first_evolution, n_iter=n_iter, stds=stds, adaptation_probability=adaptation_probability, base_growth_rate=base_growth_rate)[1])

#ratios_per_population_size = [] 
#individual_ratios_per_population_size = []
#for max_population_size in [100,500,1000,5000,10000]:
#    ratios_per_population_size.append(main(first_evolution, n_iter=n_iter, stds=stds, adaptation_probability=0.001, base_growth_rate=base_growth_rate, max_population_size=max_population_size)[0])
#    individual_ratios_per_population_size.append(main(first_evolution, n_iter=n_iter, stds=stds, adaptation_probability=0.001, base_growth_rate=base_growth_rate, max_population_size=max_population_size)[1])
#ratio_per_n_iter = []
#for n_iter in [100,500,1000,5000,10000]:
#    ratio_per_n_iter.append(main(first_evolution, n_iter=n_iter, stds=stds, adaptation_probability=0.05, base_growth_rate=base_growth_rate))


plt.figure()
plt.plot([0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1], ratios_per_probability, label="Mean variance")
plt.plot([0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1], individual_ratios_per_probability, label="Individual variance")
plt.loglog()
plt.xlabel("Adaptation probability")
plt.ylabel("Ratio of mean deviation through replicates")
plt.title("Ratio of mean deviation through replicates as a function of adaptation probability")
plt.legend()


plt.figure()
plt.plot([100,500,1000,5000,10000], ratios_per_population_size, label="Mean variance")
plt.plot([100,500,1000,5000,10000], individual_ratios_per_population_size, label="Individual variance")
plt.loglog()
plt.xlabel("Max population size")
plt.ylabel("Ratio of mean deviation through replicates")
plt.title("Ratio of mean deviation through replicates as a function of max population size")
plt.legend()


plt.show()

"""
