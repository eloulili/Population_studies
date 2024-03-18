from typing import Optional
import numpy as np
import matplotlib.scale as scle
import matplotlib.pyplot as plt

np.random.seed(0)
N = 200
MAX_CONDITIONS = 9
probabilities = [0.1, 0.01, 1.1]
first_evolution = [7]
numbers = [200]
conditions_profile = [
    (9, 1000),
    (8, 1000),
    (7, 1000),
    (6, 1000),
    (5, 1000),
    (4, 1000),
    (3, 1000),
    (2, 1000),
    (1, 1000),
    (0, 1000),
    (9, 1000),
    (8, 1000),
    (7, 1000),
    (6, 1000),
    (5, 1000),
    (4, 1000),
    (3, 1000),
    (2, 1000),
    (1, 1000),
    (0, 1000),
    (9, 1000),
    (8, 1000),
    (7, 1000),
    (6, 1000),
    (5, 1000),
    (4, 1000),
    (3, 1000),
    (2, 1000),
    (1, 1000),
    (0, 1000),
]
inherent_growth_rates = [
    0.5 / (1.1**i) for i in range(MAX_CONDITIONS + 1)
]  # assume that condition 0 is the best and condition 9 is the worst
DISTANT_COEFF = 0.1


def growth_rate(best_gene_distance, evolutions, condition):
    return (
        1 / (1 + 0.1 * best_gene_distance)
        + inherent_growth_rates[max(evolutions, condition)]
    )
    # the first term is the adaptation, the second term is the inherent growth rate that can be capted by the cell


def growth_rate(best_gene_distance, evolutions, condition):
    return np.exp(
        -(DISTANT_COEFF * best_gene_distance)
        + inherent_growth_rates[max(evolutions, condition)]
    )


class EvolutiveCells1D:
    def __init__(self, type: int, first_evolution: Optional[int] = None, conditions=0):

        self.type = type
        self.short_evolutions = (
            []
        )  # the first element is the adaptation, the second is the duration
        self.long_evolution = first_evolution
        self.growth_rate = growth_rate(
            abs(first_evolution - conditions), self.long_evolution, conditions
        )

    def get_growth_rate(self):
        return self.growth_rate

    def get_name(self):
        return self.name

    def update_cell(
        self,
        conditions,
        probability_to_evolve: float = 0.1,
        distance_mult: float = 1.1,
        probability_to_adapt: float = 0.1,
    ) -> tuple[int, int]:
        best_distance = abs(self.long_evolution - conditions)
        new_evolution = -1
        new_adaptation = -1
        if self.short_evolutions != []:
            best_adaptation = min(
                self.short_evolutions, key=lambda x: abs(x[0] - conditions)
            )
            best_distance = min(best_distance, abs(best_adaptation[0] - conditions))
        self.growth_rate = growth_rate(best_distance, self.long_evolution, conditions)
        for evolution in self.short_evolutions:
            if evolution[1] == 0:
                self.short_evolutions.remove(evolution)
            else:
                evolution[1] -= 1

            distance = abs(evolution[0] - self.long_evolution)
            if np.random.uniform(0, 1) < probability_to_evolve / (
                distance_mult**distance
            ):
                self.long_evolution = evolution[0]
                new_evolution = evolution[0]

        if (
            np.random.uniform(0, 1) < probability_to_adapt
            and self.long_evolution != conditions
        ):
            if self.long_evolution < conditions:
                new_adaptation = np.random.randint(self.long_evolution, conditions + 1)
                self.short_evolutions.append([new_adaptation, 5])
            else:
                new_adaptation = np.random.randint(conditions, self.long_evolution + 1)
                self.short_evolutions.append([new_adaptation, 5])
        return new_evolution, new_adaptation

    def copy(self, conditions):
        new_cell = EvolutiveCells1D(self.type, conditions)
        new_cell.short_evolutions = self.short_evolutions.copy()
        new_cell.long_evolution = self.long_evolution
        return new_cell

    def __str__(self):
        return f"Cell of type {self.type} has adaptation {self.short_evolutions} and long evolution {self.long_evolution}"


class EvolutiveSample1D:

    def __init__(self, cells: list[EvolutiveCells1D], nb_types: int):
        self.cells = cells
        self.n = len(cells)
        self.nb_types = nb_types
        self.cumulative_growth_rates = []
        self.conditions = None
        self.evolution_count = [0 for i in range(MAX_CONDITIONS + 1)]
        self.adaptation_count = [0 for i in range(MAX_CONDITIONS + 1)]
        cumul = 0.0
        for i in range(self.n):
            cumul += self.cells[i].growth_rate
            self.cumulative_growth_rates.append(cumul)

    def change_conditions(self, conditions):
        self.conditions = conditions
        for cell in self.cells:
            best_distance = abs(cell.long_evolution - conditions)
            if cell.short_evolutions != []:
                best_adaptation = min(
                    cell.short_evolutions, key=lambda x: abs(x[0] - conditions)
                )
                best_distance = min(best_distance, abs(best_adaptation[0] - conditions))
            cell.growth_rate = growth_rate(
                best_distance, cell.long_evolution, conditions
            )
        cumul = 0.0
        for i in range(self.n):
            cumul += self.cells[i].growth_rate
            self.cumulative_growth_rates[i]

    def get_proportions_per_type(self):
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

    def get_proportions_per_evolution(self):
        return [
            sum([cell.long_evolution == i for cell in self.cells]) / self.n
            for i in range(MAX_CONDITIONS + 1)
        ]

    def get_proportion_per_adaptation(self):
        return [
            sum(
                [
                    evolution[0] == i
                    for cell in self.cells
                    for evolution in cell.short_evolutions
                ]
            )
            / self.n
            for i in range(MAX_CONDITIONS + 1)
        ]

    def get_propotions_per_evolution_and_type(self, type: int):
        assert type < self.nb_types
        return [
            sum([cell.long_evolution == i and cell.type == type for cell in self.cells])
            / self.n
            for i in range(MAX_CONDITIONS)
        ]

    def update(
        self,
        birth_index,
        death_index,
        evolution_probabilty: float = 0.01,
        adaptation_probability: float = 0.01,
        distance_mult: float = 1.1,
    ):
        new_cell = self.cells[birth_index].copy(self.conditions)
        new_evol, new_adap = new_cell.update_cell(
            self.conditions,
            probability_to_evolve=evolution_probabilty,
            distance_mult=distance_mult,
            probability_to_adapt=adaptation_probability,
        )
        for i in range(death_index, self.n):
            self.cumulative_growth_rates[i] += (
                new_cell.growth_rate - self.cells[death_index].growth_rate
            )
        self.cells[death_index] = new_cell
        if new_evol != -1:
            self.evolution_count[new_evol] += 1
        if new_adap != -1:
            self.adaptation_count[new_adap] += 1

    def __str__(self):
        string = ""
        for cell in self.cells:
            string += str(cell) + "\n"

        return string


def Moran_process(
    sample: EvolutiveSample1D,
    conditions_profile: list[tuple[int, int]],
    evolution_probabilty: float = 0.01,
    adaptation_probability: float = 0.01,
    distance_mult: float = 1.1,
):
    proportions_type = [sample.get_proportions_per_type()]
    proportions_evolution = [sample.get_proportions_per_evolution()]
    proportion_adaptation = [sample.get_proportion_per_adaptation()]
    proportion_any_adaptation = [0]
    growth_rate_by_type = [sample.get_mean_growth_rate_by_type()]
    growth_rates = [sample.get_mean_growth_rate()]
    for conditions, n in conditions_profile:
        sample.change_conditions(conditions)
        for i in range(n):
            birth_rate = np.random.uniform(0, sample.get_cumulative_growth_rate())
            birth_index = np.searchsorted(sample.cumulative_growth_rates, birth_rate)

            death_index = np.random.randint(sample.n)
            sample.update(
                birth_index,
                death_index,
                evolution_probabilty,
                adaptation_probability,
                distance_mult,
            )
            proportions_type.append(sample.get_proportions_per_type())
            proportions_evolution.append(sample.get_proportions_per_evolution())
            growth_rate_by_type.append(sample.get_mean_growth_rate_by_type())
            growth_rates.append(sample.get_mean_growth_rate())
            proportion_adaptation.append(sample.get_proportion_per_adaptation())
            proportion_any_adaptation.append(
                sum(sample.get_proportion_per_adaptation())
            )
    print(
        f"Evolutions count:{sample.evolution_count}, Total count: {sum(sample.evolution_count)}"
    )
    print(
        f"Adaptations count:{sample.adaptation_count}, Total count: {sum(sample.adaptation_count)}"
    )

    return (
        proportions_evolution,
        proportion_adaptation,
        proportions_type,
        growth_rate_by_type,
        growth_rates,
        proportion_any_adaptation,
    )


def main(
    first_evolution,
    numbers,
    conditions_profile: list[tuple[int, int]],
    probabilities: Optional[list[float]] = None,
):
    assert len(first_evolution) == len(numbers)
    for k in range(1):
        np.random.seed(k)
        cells = []
        initial_conditions = conditions_profile[0][0]
        for i in range(len(first_evolution)):
            for j in range(numbers[i]):
                cells.append(
                    EvolutiveCells1D(i, first_evolution[i], initial_conditions)
                )
        sample = EvolutiveSample1D(
            cells,
            len(first_evolution),
        )

        if probabilities != None:
            (
                proportion_evolution,
                proportion_adaptation,
                proportions_type,
                growth_rate_by_type,
                mean_growth_rates,
                proportion_any_adaptation,
            ) = Moran_process(
                sample,
                conditions_profile,
                probabilities[0],
                probabilities[1],
                probabilities[2],
            )
        else:
            (
                proportions_type,
                proportion_evolution,
                growth_rate_by_type,
                mean_growth_rates,
            ) = Moran_process(sample, conditions_profile)
        plt.figure()
        plt.plot(proportion_evolution, label=[f"evolution {i}" for i in range(10)])
        plt.xlabel("Time")
        plt.ylabel("Proportion of cells")
        plt.title("Proportion of cells per evolution")
        plt.legend()
        plt.figure()
        plt.plot(proportion_adaptation, label=[f"evolution {i}" for i in range(10)])
        plt.xlabel("Time")
        plt.ylabel("Proportion of cells")
        plt.title("Proportion of cells per adaptation")
        plt.legend()
        plt.figure()
        plt.plot(mean_growth_rates)
        plt.xlabel("Time")
        plt.ylabel("Mean growth rate")
        plt.title("Mean growth rate")
        plt.figure()
        plt.plot(proportions_type, label=[f"Type {i}" for i in range(10)])
        plt.xlabel("Time")
        plt.ylabel("Proportion of cells")
        plt.title("Proportion of cells per type")
        plt.legend()
        plt.figure()
        plt.plot(proportion_any_adaptation)
        plt.xlabel("Time")
        plt.ylabel("Proportion of cells")
        plt.title("Proportion of cells with any adaptation")
        plt.show()


main(first_evolution, numbers, conditions_profile, probabilities)
