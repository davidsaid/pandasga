from core import TransformationColumn

import numpy as np

class LinearMapping(TransformationColumn):
    def __init__(self,
                 name,
                 genotype_column,
                 max_value=1.0,
                 min_value=0.0):
        self.name = name
        self.max_value = max_value
        self.min_value = min_value
        self.genotype_column = genotype_column
    
    def scale_coefficient(self):
        return self.max_value - self.min_value
    
    def evaluate(self, population):
        i = population.individuals
        g = self.genotype_column
        u = i[g.name] / float(g.max_value())
        i[self.name] = self.scale_coefficient() * u + self.min_value

class ColumnFunction(TransformationColumn):
    def __init__(self,
                 name,
                 columns,
                 function,
                 normalize_columns=False):
        self.name = name
        self.columns = columns
        self.function = np.vectorize(function)
        self.normalize_inputs = normalize_columns
    
    def evaluate(self, population):
        i = population.individuals
        i[self.name] = self.function(*self.get_inputs(population))
    
    def get_inputs(self, population):
        output = []
        i = population.individuals
        for col in self.columns:
            if self.normalize_inputs:
                c = self.normalize(i[col])
            else:
                c = i[col] 
            output.append(c)
        return output

    def normalize(self, column):
        c = column - column.min()
        return c / c.sum()