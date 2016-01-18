from core import GeneticOperator

import numpy as np

class StochasticUniformSelection(GeneticOperator):
    def __init__(self,
                column,
                maximize=True):
        self.column = column
        self.maximize = maximize
    
    def iterate(self, population):
        N = 2*population.generation_size
        delta = 1.0/N
        offset = delta*np.random.rand()
        ticks = [delta*i + offset for i in xrange(N)]
        
        data = population.individuals[self.column]
        shifted_data = data - data.min()
        cdf = (shifted_data / shifted_data.sum()).cumsum()
        if not self.maximize:
            cdf = 1 - cdf
        
        population.mating_pool = [np.argmax(cdf >= tick) for tick in ticks]

class KTournamentSelection(GeneticOperator):
    def __init__(self, \
                 tournament_size,
                 column,
                 maximize=True):
        self.tournament_size = tournament_size
        self.column = column
        self.maximize = maximize
    
    def iterate(self, population):
        population.mating_pool = [self.tournament(population) for _ in xrange(2 * population.generation_size)]

    def tournament(self, population):
        contender_indices = np.random.choice(len(population), self.tournament_size)
        contenders = population.individuals.loc[contender_indices]
        
        if self.maximize:
            return contenders[self.column].idxmax()
        else:
            return contenders[self.column].idxmin()
        
class SelectWorstNLethals(GeneticOperator):
    def __init__(self, column, maximize=True):
        self.column = column
        self.maximize = maximize
    
    def iterate(self, population):
        index = population.individuals.sort_values(by=self.column, ascending=self.maximize).index
        population.survivor_list = index[population.generation_size:]
        population.lethal_list = index[:population.generation_size]