# Import necessary libraries for typing annotations, mathematical operations, plotting, profiling, and timing.
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from pstats import Stats
import cProfile
import time
import statistics
from scipy.stats import differential_entropy as entropy


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
PEAKS = None
CHANGE_PEAK_PROBABILITY = 0.03

def compute_next_time_step(n: int, base_growth_rate: float, type:str, variance:float = 0.05) -> float:
    if type == "lognormal":
        mu = np.log(base_growth_rate / np.sqrt(1 + variance / base_growth_rate ** 2))
        sigma = np.sqrt(np.log(1 + variance / base_growth_rate ** 2))
        return np.random.lognormal(mu , sigma)
    if type == "exponential":
        return np.random.exponential(1 / (n * base_growth_rate))

def compute_new_phenotype( mother_cell, stds: list[float], probability_to_adapt: float) -> list[float]:   
    """
    Compute the new phenotype state of a cell based on its mother cell's phenotype state and the standard deviations for each trait.
    If the distribution of traits is multimodal, the new phenotype state may be chosen from a set of peaks.
    Feel free to modify this function to include additional logic or constraints as needed.
    """
    new_phenotype = mother_cell.phenotype.copy()
    new_peak = mother_cell.associated_peak.copy()
    if np.random.uniform() < probability_to_adapt:
                    new_phenotype =  mother_cell.phenotype.copy()
                    new_phenotype = list(np.random.normal(mother_cell.phenotype, stds))
    if PEAKS is not None:
                if np.random.uniform() < CHANGE_PEAK_PROBABILITY:
                    new_peak = PEAKS[np.random.randint(0, len(PEAKS))]
                    new_phenotype =  new_peak
                    new_phenotype = list(np.random.normal(new_peak, stds))     
    return new_phenotype, new_peak                     
    
            



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
    def __init__(self, type: int, phenotype: list[float]):
        self.type = type
        self.phenotype = phenotype
        self.absolute_generation = 0
        self.previous_phenotype = [self.phenotype]
        self.associated_peak = None
        self.growth_rate = 0.5

    # Methods for updating cell state, copying the cell, and converting cell information to a string are omitted for brevity.

    # More classes and functions such as EvolutiveSample1D, Moran_process, and main are defined here.
    # These functions handle simulation setup, execution, and data analysis, including adapting to new conditions, updating cell states, and visualizing results.

    def reproduce(self, probability_to_adapt: float = 0.1,
        stds = list[float]):

        assert len(stds) == len(self.phenotype)
        new_cell = EvolutiveCell(self.type, phenotype=self.phenotype)
        new_cell.absolute_generation = self.absolute_generation +1
        new_cell.previous_phenotype = self.previous_phenotype.copy()
        new_phenotype, new_peak = compute_new_phenotype(self, stds, probability_to_adapt)
        new_cell.phenotype = new_phenotype
        new_cell.associated_peak = new_peak
        if new_phenotype != self.phenotype:
            new_cell.previous_phenotype.append(new_phenotype)
        return new_cell
        


class EvolutiveSample:

    def __init__(self, cells: list[EvolutiveCell], nb_types: int):
        self.cells = cells
        self.n = len(cells)
        self.cumulative_growth_rates = [] 

        self.list_evolutions = [cell.phenotype for cell in self.cells if cell.phenotype is not None]  

        # Variables used for tracking and analysis
        self.sum_absolute_generation = 0
        self.sum_evolution = cells[0].phenotype.copy()
        for i in  range(1, len(cells)):
            for j in range(len(cells[i].phenotype)):
                self.sum_evolution[j] += cells[i].phenotype[j]
        
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
        
        # Update the sample by adding a new cell based on the birth index and adaptation probability.
        new_cell = self.cells[birth_index].reproduce(adaptation_probability, stds)

        evol_new_cell = new_cell.phenotype 
        
        self.genetic_tree.append(new_cell.previous_phenotype + [new_cell.phenotype])

        self.sum_absolute_generation += new_cell.absolute_generation


        if self.nb_types > 1:
            self.quantity_per_type[new_cell.type] += 1

        for i in range(len(evol_new_cell)):
            self.sum_evolution[i] += evol_new_cell[i]
        self.list_evolutions.append(evol_new_cell)
        self.cells.append(new_cell)
        self.n += 1

    
    def get_ascendance(self, list_evolution):
        """Get the ascendance of a list of evolutions."""
        tree = []
        for evolution in list_evolution:
            for tree_evolution in self.genetic_tree:
                if evolution == tree_evolution[-1]:
                    tree.append((evolution, tree_evolution))
                    break
        return tree



def Gillespie_function(
    sample: EvolutiveSample,
    adaptation_probability: float = 0.01,
    base_growth_rate: float = 0.5,
    stds = list[float],
    max_population_size: int = N
):
    """
    Simulate the Moran process with adaptation and loss of adaptation.
    To compute the next time step, the growth rate is drawn from a distribution chosen by the operator 
    like in a gillespie algorithm.
    """

    quantity_type = [sample.quantity_per_type.copy()]
    absolute_generation = [sample.sum_absolute_generation/sample.n]
    mean_phenotype = [[sample.sum_evolution[i]/sample.n for i in range(len(sample.sum_evolution))]]
    current_time = 0
    timesteps = [0]
    n_timesteps = 1
    division_time = []
    while sample.n < max_population_size:
            # Compute the next time step and update the sample.
            next_time = compute_next_time_step(sample.n, base_growth_rate, "lognormal") / sample.n
            current_time += next_time
            division_time.append(next_time * sample.n)
            
            #Chose randomly a cell to reproduce
            birth_index = np.random.randint(sample.n)


            sample.update(
                birth_index = birth_index,
                adaptation_probability = adaptation_probability,
                stds = stds,
            )

            timesteps.append(current_time)
            absolute_generation.append(sample.sum_absolute_generation/sample.n)
            quantity_type.append(sample.quantity_per_type.copy())
            mean_phenotype.append([sample.sum_evolution[i]/sample.n for i in range(len(sample.sum_evolution))])

            n_timesteps += 1  

    transpose_quantity_type = [[] * sample.nb_types] 
    """""
    # If we want to use the quantity per type
    for i in range(len(quantity_type)):
        for j in range(sample.nb_types):
            if len(transpose_quantity_type) <= j:
                break
            transpose_quantity_type[j].append(quantity_type[i][j])
    """
    mean_phenotype = np.array(mean_phenotype).T

    return (
        timesteps,
        transpose_quantity_type,
        mean_phenotype,
        absolute_generation,
        division_time,
    )


def main(
    first_evolution,
    adaptation_probability: float = 0.01,
    base_growth_rate: float = 0.5,
    stds = list[float],
    n_iter: int = 1,
    max_population_size: int = N
):
    
    # Compare the variance of the population of cells in one chamber with the variance of 
    # the population of cells in all chambers, when the original cell is the same for all chambers.
    np.random.seed(0)
    cells = []
    cells = [EvolutiveCell( 0, phenotype=first_evolution[i]) for i in range(len(first_evolution))]
    list_evolutions = []
    variance_deviation_per_simulation = []
    start = time.time()
    for k in range(n_iter):
        # Initialization, control the seed for reproducibility
        np.random.seed(k)
        cells = []
        cells = [EvolutiveCell( 0, phenotype=first_evolution[i]) for i in range(len(first_evolution))]


        sample = EvolutiveSample(
            cells,
            len(first_evolution),
        )
        
        (timesteps,
        quantity_type,
        mean_phenotype,
        absolute_generation,
        division_time,
            ) = Gillespie_function(
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
        """

        plt.figure()
        plt.plot(timesteps, absolute_generation, label="Mean absolute generation")
        plt.xlabel("Time")
        plt.ylabel("Mean absolute generation")
        plt.title("Mean absolute generation")


        plt.figure()
        plt.plot(timesteps, mean_phenotype[0], label="Mean phenotype")
        plt.xlabel("Time")
        plt.ylabel("Mean phenotype")
        plt.title("Mean phenotype")
        plt.legend()


        plt.figure()
        plt.hist(division_time, bins=100)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.title("Division time")

   

        plt.show()

      """  
        (       timesteps,
                list_evolution,
                absolute_generation,
                mean_phenotype,
                sample,
                total_genetic_tree,
                division_time,
            ) = None, None, None, None, None, None, None
    end = time.time()

    list_deviation_replicates = []
    evolution_first_std = []
    entropies = []
    for i in range(len(stds)):
        evolutions_through_replicates = []
        for j in range(n_iter):
            for k in range(max_population_size):
                evol = list_evolutions[j][i][k]
                if evol != 0:
                    evolutions_through_replicates.append(list_evolutions[j][i][k])
        list_deviation_replicates.append(np.var(evolutions_through_replicates))
        entropies.append(entropy(evolutions_through_replicates))
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

    """
    We make a population of cells grow from one cell over n_1 generations.
    We keep dilution_1 cells and make them grow over n_2 generations seperately.
    We keep dilution_2 cells and we compare the variance of the population of cells with the variance of the dilution_2 cells.
    """

    np.random.seed(seed)
    list_evolutions = []
    variance_deviation_per_simulation = []
    entropy_per_simulation = []
    std_per_simulation = []
    mean_per_simulation = []
    start = time.time()

    cells = [EvolutiveCell( 0, [0])]
    cells[0].associated_peak = [0]

    otiginal_sample = EvolutiveSample(
            cells,
            len(first_evolution),
        )
    max_population_size = 2**n_1  
    # First growth
    (timesteps,
        quantity_type,
        mean_phenotype,
        absolute_generation,
        divisions_time,
            ) = Gillespie_function(
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
        # We take a random cell from the first population and make it grow over n_2 generations
        random_index = np.random.randint(otiginal_sample.n)
        original_cell = otiginal_sample.cells[random_index]
        new_sample = EvolutiveSample(cells = [original_cell], nb_types = 1)
        (timesteps, 
        quantity_type,
        mean_phenotype, 
        absolute_generation, 
        divisions_time) = Gillespie_function(new_sample, adaptation_probability= adaptation_probability, base_growth_rate= base_growth_rate, stds = stds, max_population_size = max_population_size)
         

        evolutions = (np.array(new_sample.list_evolutions).T)
        random_indexes = np.random.randint(new_sample.n, size = int(dillution_2))
        list_evolutions.append(evolutions[0][random_indexes])
        mean_per_simulation.append(np.mean(evolutions[0]))
        if i / dillution_1 > trigger and verbose:
            print(f"Progress: {trigger * 100} %, Time elapsed: {time.time() - start}")
            trigger += 0.05
    total_populations = []
    for k in range(len(list_evolutions)):
        entropy_per_simulation.append(entropy(list_evolutions[k]))
        variance_deviation_per_simulation.append(np.var(list_evolutions[k]))
        for j in range(len(list_evolutions[k])):
            total_populations.append(list_evolutions[k][j])
    total_variance = np.var(total_populations)
    total_entropy = entropy(total_populations)
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
    expected_ratio_v2 = 1 + var_M * (n_2 - 1) * np.mean(variance_inv) # There are 2 expressions for the expected ratio, this is the second one
    effective_ratio = total_variance / np.mean(variance_deviation_per_simulation, axis=0)
    std_ratio = np.sqrt(CV_2)*(n_2 -1)/n_1
    
    print(f" CV² :{CV_2}")
    print(f"Mean_expected_ratio : {expected_ratio }")
    print(f"Ecpected ratio v2 : {expected_ratio_v2 }")
    print(f"STD on Ratio : {std_ratio}")
    #print(f"Ratio of mean deviation through replicates: {np.mean(variance_deviation_per_simulation, axis=0)/total_variance}")
    print(f"Effective Ratio: {effective_ratio}")
    print(f"Is in the range: {expected_ratio - 1.96*std_ratio<= effective_ratio  and effective_ratio <= expected_ratio + 1.96*std_ratio}")

    print(f"Mean entropy: {np.mean(entropy_per_simulation)}")
    print(f"Total entropy: {total_entropy}")
    print(f"Ratio entropy: {total_entropy/np.mean(entropy_per_simulation)}")
    if np.random.uniform() < 0.: 
        # Plot the distribution of the population sometimes
        plt.hist(total_populations, bins=100)
        plt.show()


    print(f"Parameters: stds =  {stds}, adaptation_probability = {adaptation_probability}, growth_rate = {base_growth_rate}, n_1 = {n_1}, dillution_1 = {dillution_1}, n_2 = {n_2}, dillution_2 = {dillution_2} ")

    return (expected_ratio_v2 - 1.96*std_ratio<= effective_ratio  and effective_ratio <= expected_ratio_v2 + 1.96*std_ratio,expected_ratio - 1.96*std_ratio<= effective_ratio  and effective_ratio <= expected_ratio + 1.96*std_ratio, effective_ratio, 
            expected_ratio, expected_ratio_v2, std_ratio, total_variance, CV_2, total_entropy, np.mean(entropy_per_simulation), total_entropy/np.mean(entropy_per_simulation))





n_in_range = 0
n_in_range_v2 = 0
sum_ratio = 0
sum_ratio_entropy = 0
sum_cv = 0  
sum_total_variance = 0
sum_total_entropy = 0
for seed in range(100):
    is_in_range_v2,is_in_range, effective_ratio, expected_ratio, expected_ratio_v2, std, total_variance, cv_2, total_entropy, mean_entropy, ratio_entropy = main_lior(adaptation_probability= .8, base_growth_rate=base_growth_rate, stds = [0.15], n_1=17, dillution_1= 250, n_2=9, dillution_2=256, seed=seed, verbose=False) 
    print(seed)
    sum_ratio += effective_ratio
    sum_ratio_entropy += ratio_entropy
    sum_cv += np.sqrt(cv_2)
    sum_total_variance += total_variance
    sum_total_entropy += total_entropy
    if is_in_range:
        n_in_range += 1

    if is_in_range_v2:
        n_in_range_v2 += 1
print(f"Number of simulations in range: {n_in_range}")
print(f"Percentage of simulations in range: {100 * n_in_range/100}%")
print(f"Number of simulations in range v2: {n_in_range_v2}")
print(f"Percentage of simulations in range v2: {100 * n_in_range_v2/100} %")
print(f"Mean effective ratio: {sum_ratio/100}")
print(f"Mean cv: {sum_cv/100}")
print(f"Mean total variance: {sum_total_variance/100}\n")
print(f"Mean total entropy: {sum_total_entropy/100}")
print(f"Mean ratio entropy: {sum_ratio_entropy/100}")

