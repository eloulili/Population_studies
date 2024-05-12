import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
CURRENT_MAX_TYPE = 0  # Initial maximum type
MAX_CONDITIONS = 10  # Assuming maximum 10 different types
STD = 0.1  # Standard deviation for evolution
total_time = 100  # Total simulation time
evolution_probability = 0.1

condition_changes = [(10, 2), (20, 5)]  # Scheduled changes in conditions

def growth_rate(best_gene_distance, evolutions, condition):
    return (
        1 / (1 + 0.1 * best_gene_distance)
        + 0.5 / (1.1**min(evolutions, MAX_CONDITIONS - 1))
    )

class EvolutiveCells1D:
    def __init__(self, cell_type: int, evolution : float, conditions=0, generation=0):
        self.type = cell_type
        self.evolution = evolution
        self.conditions = conditions
        self.growth_rate = growth_rate(0, self.evolution, conditions)
        self.generation = generation

    def reproduce(self, conditions):
        new_evolution = self.evolution
        if np.random.rand() < evolution_probability:
            new_type = CURRENT_MAX_TYPE +1
            new_evolution = self.evolution + np.random.normal(0, STD)
            CURRENT_MAX_TYPE += 1
        new_type = np.random.choice([self.type])  # Evolution choice could be expanded
        return EvolutiveCells1D(new_type, new_evolution, conditions, self.generation + 1)

    def update_growth_rate(self, new_conditions):
        self.conditions = new_conditions
        self.growth_rate = growth_rate(0, self.evolution, new_conditions)

class EvolutiveSample1D:
    def __init__(self, initial_conditions, nb_types: int):
        self.cells = [EvolutiveCells1D(0, initial_conditions) for _ in range(200)]  # All start with the initial type 0
        self.nb_types = nb_types
        self.current_time = 0
        self.conditions = initial_conditions
        self.total_rate = sum(cell.growth_rate for cell in self.cells)
        self.proportions = np.zeros(nb_types)
        self.proportions[0] = 1.0  # All cells are initially of type 0
        self.history = {'time': [], 'proportions': []}

    def simulate(self, total_time):
        next_condition_change = condition_changes.pop(0) if condition_changes else (None, None)
        while self.current_time < total_time:
            if next_condition_change[0] and self.current_time >= next_condition_change[0]:
                self.update_conditions(next_condition_change[1])
                if condition_changes:
                    next_condition_change = condition_changes.pop(0)
                else:
                    next_condition_change = (None, None)
            time_step = np.random.exponential(1 / self.total_rate)
            self.current_time += time_step
            if self.current_time > total_time:
                break
            self.perform_reproduction_and_replacement()
            self.history['time'].append(self.current_time)
            self.history['proportions'].append(self.proportions.copy())

    def update_conditions(self, new_conditions):
        self.conditions = new_conditions
        for cell in self.cells:
            cell.update_growth_rate(new_conditions)

    def perform_reproduction_and_replacement(self):
        event_rate = np.random.uniform(0, self.total_rate)
        selected_index = np.searchsorted(np.cumsum([cell.growth_rate for cell in self.cells]), event_rate)
        new_cell = self.cells[selected_index].reproduce(self.conditions)
        death_index = np.random.randint(len(self.cells))
        death_type = self.cells[death_index].type
        new_type = new_cell.type

        # Update proportions
        self.proportions[death_type] -= 1 / len(self.cells)
        self.proportions[new_type] += 1 / len(self.cells)

        # Replace the cell
        self.total_rate -= self.cells[death_index].growth_rate
        self.cells[death_index] = new_cell
        self.total_rate += new_cell.growth_rate

    def display_results(self):
        time_points = np.array(self.history['time'])
        for type_index in range(self.nb_types):
            plt.plot(time_points, np.array(self.history['proportions'])[:, type_index], label=f'Type {type_index}')
        plt.xlabel('Time')
        plt.ylabel('Proportion of Each Type')
        plt.title('Evolution of Cell Types Over Time')
        plt.legend()
        plt.show()

def main():
    initial_conditions = 0
    nb_types = MAX_CONDITIONS
    sample = EvolutiveSample1D(initial_conditions, nb_types)
    sample.simulate(total_time)
    sample.display_results()

main()