import numpy as np
import matplotlib.pyplot as plt


class Cell:
    def __init__(self, growth_rate: float, type: int):
        self.growth_rate = growth_rate
        self.type = type

    def get_growth_rate(self):
        return self.growth_rate

    def get_name(self):
        return self.name

    def __str__(self):
        return f"Cell of type{self.name} has growth rate {self.growth_rate}"


class Sample:

    def __init__(self, cells: list[Cell], nb_types: int):
        self.cells = cells
        self.n = len(cells)
        self.nb_types = nb_types
        self.cumulative_growth_rates = []
        cumul = 0.0
        for i in range(self.n):
            cumul += self.cells[i].growth_rate
            self.cumulative_growth_rates.append(cumul)

    def get_proportions(self):
        return [
            sum([cell.type == i for cell in self.cells]) / self.n
            for i in range(self.nb_types)
        ]

    def get_mean_growth_rate(self):
        return sum([cell.growth_rate for cell in self.cells]) / self.n

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

    def get_cumulative_growth_rate(self):
        return self.cumulative_growth_rates[-1]

    def update(self, birth_index, birth_growth_rate, death_index):
        new_cell = Cell(birth_growth_rate, self.cells[birth_index].type)
        for i in range(death_index, self.n):
            self.cumulative_growth_rates[i] += (
                new_cell.growth_rate - self.cells[death_index].growth_rate
            )
        self.cells[death_index] = new_cell


def Moran_process(sample: Sample, n: int, possible_relative_error: float = 0.01):
    proportions = [sample.get_proportions()]
    growth_rate_by_type = [sample.get_mean_growth_rate_by_type()]
    growth_rates = [sample.get_mean_growth_rate()]
    for i in range(n):
        birth_rate = np.random.uniform(0, sample.get_cumulative_growth_rate())
        birth_index = np.searchsorted(sample.cumulative_growth_rates, birth_rate)
        new_growth_rate = sample.cells[birth_index].growth_rate * np.random.uniform(
            1 - possible_relative_error, 1 + possible_relative_error
        )
        death_index = np.random.randint(sample.n)
        sample.update(birth_index, new_growth_rate, death_index)
        proportions.append(sample.get_proportions())
        growth_rate_by_type.append(sample.get_mean_growth_rate_by_type())
        growth_rates.append(sample.get_mean_growth_rate())
    return proportions, growth_rate_by_type, growth_rates


def main(growth_rates, numbers, N=10000, error=0.03):
    assert len(growth_rates) == len(numbers)
    cells = []
    for i in range(len(growth_rates)):
        for j in range(numbers[i]):
            cells.append(
                Cell(
                    growth_rates[i],
                    i,
                )
            )
    sample = Sample(cells, len(growth_rates))
    proportions, growth_rate_by_type, mean_growth_rates = Moran_process(
        sample, N, error
    )
    growth_rate_evolution_per_type = np.array(growth_rate_by_type).T
    growth_rate_evolution_per_type = growth_rate_evolution_per_type - np.array(
        mean_growth_rates
    )
    plt.figure()
    plt.plot(proportions)
    plt.show()
    plt.figure()
    plt.plot(growth_rate_evolution_per_type.T)
    plt.show()
    plt.figure()
    plt.plot(mean_growth_rates)
    plt.show()


growth_rate = [2.2, 1.9, 2.1, 2.5]
numbers = [200, 200, 200, 10]
main(growth_rate, numbers)
