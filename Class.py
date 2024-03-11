


class Cell:
    def __init__(self, growth_rate, name):
        self.growth_rate = growth_rate
        self.name = name

    def get_growth_rate(self):
        return self.growth_rate

    def get_name(self):
        return self.name

    def __str__(self):
        return f"Cell of type{self.name} has growth rate {self.growth_rate}"
    

class Sample:

    def __init__(self, cells):
        self.cells = cells
        self.n = len(cells)

    def add_cell(self, cell):
        self.cells.append(cell)

    def get_cells(self):
        return self.cells

    def get_name(self):
        return self.name

    def __str__(self):
        return f"Sample {self.name} has {len(self.cells)} cells"