from core import GeneticOperator

import numpy as np

class KTournamentSelection(GeneticOperator):
    def __init__(self, \
                 tournament_size,
                 axis=0):
        self.tournament_size = tournament_size
        self.axis=0
    
    def iterate(self, population):
        population.mating_pool = [self.tournament(population) for _ in xrange(2 * population.generation_size)]

    def tournament(self, population):
        contender_indices = np.random.choice(len(population), self.tournament_size)
        contenders = population.individuals.loc[contender_indices]
        
        target = population.targets[self.axis]
        if target.maximization_flag:
            return contenders[target.name].idxmax()
        else:
            return contenders[target.name].idxmin()
        
class SelectWorstNLethals(GeneticOperator):
    def __init__(self, axis=0):
        self.axis=0
    
    def iterate(self, population):
        target = population.targets[self.axis]
        index = population.individuals.sort_values(by=target.name, ascending=target.maximization_flag).index
        population.lethal_list = index[:population.generation_size]
        population.survivor_list = index[population.generation_size:]