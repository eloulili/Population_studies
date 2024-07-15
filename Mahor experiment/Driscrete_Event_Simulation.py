import random
import numpy as np
import heapq
import matplotlib.pyplot as plt
import time
import cProfile


# Set the seed for NumPy's random number generator to ensure reproducibility.
np.random.seed(0)

# Define constants and simulation parameters.
ONLY_DAUGHTER_EVOLVES = False # If False, the mother may also evolve

adaptation_probability = 1  # Probability of adaptation
base_growth_rate = 0.5  # Base growth rate
std = 0.1  # Standard deviations for the evolution
std_time = 0.1  # Standard deviation for the time between divisions



def next_interval(rate, variance, type):
    if type == "lognormal":
        mu = np.log(1 /(rate  * np.sqrt(1 + variance * rate**2)))
        sigma = np.sqrt(np.log(1 + variance * rate**2))
        return np.random.lognormal(mu, sigma)
    if type == "exponential":
        return np.random.exponential(1 / (rate))
    if type == "normal":
        return np.random.normal(1/rate, variance)



def compute_new_phenotype(
    mother_cell,std,  probability_to_adapt: float
) -> list[float]:
    """
    Compute the new phenotype state of a cell based on its mother cell's phenotype state and the standard deviations for each trait.
    If the distribution of traits is multimodal, the new phenotype state may be chosen from a set of peaks.
    Feel free to modify this function to include additional logic or constraints as needed.
    """
    new_phenotype = mother_cell.phenotype
    if np.random.uniform() < probability_to_adapt:
        new_phenotype = np.random.normal(float(mother_cell.phenotype), std)
    return new_phenotype

class EvolutiveCell:
    """Class to represent an evolutionary cell with specific characteristics and behaviors."""

    def __init__(self,  phenotype: float, num_cell: int = 0):
        self.phenotype = phenotype
        self.absolute_generation = 0
        self.growth_rate = 0.5
        self.num_cell = num_cell

    # Methods for updating cell state, copying the cell, and converting cell information to a string are omitted for brevity.

    # More classes and functions such as EvolutiveSample1D, Moran_process, and main are defined here.
    # These functions handle simulation setup, execution, and data analysis, including adapting to new conditions, updating cell states, and visualizing results.

    def reproduce(self, probability_to_adapt: float = 0.1, std: float = std):

        new_cell = EvolutiveCell( phenotype=self.phenotype)
        new_cell.absolute_generation = self.absolute_generation + 1
        new_phenotype = compute_new_phenotype(
            self, std,  probability_to_adapt
        )
        new_cell.phenotype = new_phenotype

        if not ONLY_DAUGHTER_EVOLVES:
            new_phenotype = compute_new_phenotype(
                self, std, probability_to_adapt
            )

            self.phenotype = new_phenotype
            
        return new_cell


class System(object):
    
    def __init__(self, initial_cell, adaptation_probability = 0.1, max_population = 2**18):
        self.schedule = []
        self.cells = [initial_cell]
        self.division_times = []
        self.n_cells = 1
        self.n_next_divisions = 0
        self.adaptation_probability = adaptation_probability
        self.max_time_schedule = 0.
        self.max_population = max_population


class Event(object):
    
    def __init__(self, time):
        self.time = time
        self.type = ""
    

    
class Division(Event):
    def __init__(self, time, cell ):
        super().__init__(time)
        self.type = "division"
        self.cell = cell
        self.new_division_times = []

    def action(self, system):
        new_cell = self.cell.reproduce(system.adaptation_probability)
        new_cell.num_cell = system.n_cells 
        system.cells.append(new_cell)
        system.cells[self.cell.num_cell] = self.cell
        system.n_cells += 1
        system.n_next_divisions -= 1

        next_division_time_mother = next_interval(self.cell.growth_rate, std_time, "normal") + self.time
        if system.n_cells + system.n_next_divisions < system.max_population +2 or next_division_time_mother < system.max_time_schedule:
            division_event_mother = Division(next_division_time_mother, self.cell)
            heapq.heappush(system.schedule, (next_division_time_mother , division_event_mother))
            system.n_next_divisions += 1
            if next_division_time_mother > system.max_time_schedule:
                system.max_time_schedule = next_division_time_mother
            if system.n_cells + system.n_next_divisions  >= system.max_population +2:
                heapq.heappop(system.schedule)
                system.max_timeschedule = system.schedule[-1][0]
                system.n_next_divisions -= 1

        next_division_time_daughter = next_interval(new_cell.growth_rate, std_time, "normal") + self.time
        if system.n_cells + system.n_next_divisions <system.max_population +2 or next_division_time_daughter < system.max_time_schedule:
            division_event_mother = Division(next_division_time_daughter, new_cell)
            heapq.heappush(system.schedule, (next_division_time_daughter , division_event_mother))
            system.n_next_divisions += 1
            if next_division_time_daughter > system.max_time_schedule:
                system.max_time_schedule = next_division_time_daughter
            if system.n_cells + system.n_next_divisions >= system.max_population +2 :
                heapq.heappop(system.schedule)
                system.max_timeschedule = system.schedule[-1][0]
                system.n_next_divisions -= 1




def main(n_1: int, n_2 : int, dillution : int, adaptation_probability: float):
    initial_cell = EvolutiveCell(phenotype= 10)
    system = System(initial_cell, adaptation_probability, max_population = 2**n_1)
    next_division_time = next_interval(initial_cell.growth_rate, std_time, "normal")
    division_event = Division(next_division_time, initial_cell)
    system.max_time_schedule = next_division_time
    system.n_next_divisions = 1
    heapq.heappush(system.schedule, (next_division_time, division_event))
    while system.n_cells <system.max_population:
        time, event = heapq.heappop(system.schedule)
        event.action(system)


    chosen_cells = random.sample(system.cells, dillution)
    evolutions = []
    for cell in chosen_cells:
        cell.num_cell = 0
        system = System(cell, adaptation_probability, max_population = 2**n_2)
        next_division_time = next_interval(cell.growth_rate, std_time, "normal")
        division_event = Division(next_division_time, cell)
        heapq.heappush(system.schedule, (next_division_time, division_event))
        while system.n_cells <system.max_population:
            time, event = heapq.heappop(system.schedule)
            event.action(system)
        phenotypes = [cell.phenotype for cell in system.cells]
        evolutions.append(phenotypes)
    
    variances_per_simulation = []
    evolution_total_population = []
    for evolution in evolutions:
        variances_per_simulation.append(np.var(evolution))
        evolution_total_population.extend(evolution)
    
    if np.random.uniform() < 0.05:
        plt.hist(evolution_total_population, bins=1000)
        plt.show()
    variance_total_population = np.var(evolution_total_population)
    variance_inv = 1 / np.array(variances_per_simulation)
    var_M = adaptation_probability *  std**2
    CV_2 = np.var(variances_per_simulation) / (np.mean(variances_per_simulation)**2)

    expected_ratio = 1 + ((n_2-1) / (n_1) )* (1 + CV_2)
    expected_ratio_v2 = 1 + var_M * (n_2 - 1) * np.mean(
        variance_inv
    )  # There are 2 expressions for the expected ratio, this is the second one
    effective_ratio = variance_total_population / np.mean(
        variances_per_simulation, axis=0
    )

    std_ratio = np.sqrt(CV_2) * (n_2 - 1) / n_1

    return (
        expected_ratio_v2 - 1.96 * std_ratio <= effective_ratio
        and effective_ratio <= expected_ratio_v2 + 1.96 * std_ratio,
        expected_ratio - 1.96 * std_ratio <= effective_ratio
        and effective_ratio <= expected_ratio + 1.96 * std_ratio,
        effective_ratio,
        expected_ratio,
        expected_ratio_v2,
        std_ratio,
        variance_total_population,
        CV_2,
    )


total_in_range = []
total_in_range_v2 = []
total_ratio = []
total_expected_ratio = []
total_expected_ratio_v2 = []
N_SIM = 50
for adaptation_probability in [0.001, 0.005, 0.01, 0.05]:
    n_in_range = 0
    n_in_range_v2 = 0
    sum_ratio = 0
    sum_cv = 0
    sum_total_variance = 0
    sum_expected_ratio = 0
    sum_expected_ratio_v2 = 0
    sum_std_ratio = 0
    start = time.time()
    for seed in range(N_SIM):
        (
            is_in_range_v2,
            is_in_range,
            effective_ratio,
            expected_ratio,
            expected_ratio_v2,
            std,
            total_variance,
            cv_2,
        ) = main(n_1 = 18, n_2 = 9, dillution = 1000, adaptation_probability = adaptation_probability

        )
        print(seed)
        sum_ratio += effective_ratio
        sum_cv += np.sqrt(cv_2)
        sum_total_variance += total_variance
        sum_expected_ratio += expected_ratio
        sum_expected_ratio_v2 += expected_ratio_v2
        sum_std_ratio += std
        if is_in_range:
            n_in_range += 1

        if is_in_range_v2:
            n_in_range_v2 += 1
    end = time.time()
    print(f"Time: {end - start}\n")
    total_in_range.append(n_in_range*2)
    total_in_range_v2.append(n_in_range_v2*2)
    total_ratio.append(sum_ratio/N_SIM)
    total_expected_ratio.append(sum_expected_ratio/N_SIM)
    total_expected_ratio_v2.append(sum_expected_ratio_v2/N_SIM)

    print(f"Adaptation probability: {adaptation_probability}")
    print(f"Number of simulations in range: {n_in_range}")
    print(f"Percentage of simulations in range: {100 * n_in_range/N_SIM}%")
    print(f"Number of simulations in range v2: {n_in_range_v2}")
    print(f"Percentage of simulations in range v2: {100 * n_in_range_v2/N_SIM} %")
    print(f"Mean effective ratio: {sum_ratio/N_SIM}")
    print(f"Mean std ratio: {sum_std_ratio/N_SIM}")
    print(f"Mean expected ratio: {sum_expected_ratio/N_SIM}")
    print(f"Mean expected ratio v2: {sum_expected_ratio_v2/N_SIM}")
    print(f"Mean cv: {sum_cv/N_SIM}")
    print(f"CV(alpha): {sum_std_ratio/sum_ratio}")
    print(f"Mean total variance: {sum_total_variance/N_SIM}\n")
    
print(total_expected_ratio)
print(total_expected_ratio_v2)
print(total_ratio)
print(total_in_range)
print(total_in_range_v2)
