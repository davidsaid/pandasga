from core import GeneticOperator
from operators import Evaluation
import numpy.matlib
from pandasga.core import GeneticOperator
from astropy.io.ascii.core import Column

def phenotype_distance(population):
    d_square = 0
    N = len(population)
    for column in population.phenotype:
        x = numpy.matlib.repmat(population.individuals[column.name], N, 1)
        y = numpy.transpose(x)
        d_square += numpy.square(x - y)
    return numpy.sqrt(d_square)

def new_linear_decay_sharing(radius):
    return lambda distance: linear_decay_sharing(distance, radius)

def linear_decay_sharing(distance, r):
    o = 0.0
    d = abs(distance)
    if r > 0 and d <= r:
        o = 1.0 - d/r
    return o

class NicheSharingCoefficient(GeneticOperator):
    def __init__(self, name, distance_function, sharing_function):
        self.name = name
        self.distance_function = distance_function
        self.sharing_function = numpy.vectorize(sharing_function)
        self.sharing_matrix = None
    
    def iterate(self, population):
        self.update_matrix(population)
        population.individuals[self.name] =\
            numpy.sum(self.sharing_matrix, axis=1)
    
    def initialize(self, population):
        self.iterate(population)
    
    def update_matrix(self, population):
        d = self.distance_function(population)
        self.sharing_matrix = self.sharing_function(d)

class MultimodalEvaluation(Evaluation):
    def __init__(self, sharing_column, target_column, output_column):
        self.sharing_column = sharing_column
        self.target_column = target_column
        self.output_column = output_column
    
    def apply(self, population):
        Evaluation.apply(self, population)
        individuals = population.individuals
        norm = self.normalize(individuals[self.target_column])
        individuals[self.output_column] =\
            norm/individuals[self.sharing_column]
    
    def normalize(self, column):
        shift = column.min()
        scale = column.max() - column.min()
        return (column - shift) / scale

class SelectWorstNLethals(GeneticOperator):
    def __init__(self, column, maximization_flag):
        self.column = column
        self.maximization_flag = maximization_flag
    
    def iterate(self, population):
        index = population.individuals.sort_values(by=self.column,\
                                                   ascending=self.maximization_flag).index
        population.lethal_list = index[:population.generation_size]
        population.survivor_list = index[population.generation_size:]
        
class KTournamentSelection(GeneticOperator):
    def __init__(self, \
                 tournament_size,
                 column,
                 maximization_flag):
        self.tournament_size = tournament_size
        self.column=column
        self.maximization_flag=maximization_flag
    
    def iterate(self, population):
        population.mating_pool = [self.tournament(population) for _ in xrange(2 * population.generation_size)]

    def tournament(self, population):
        contender_indices = numpy.random.choice(len(population), self.tournament_size)
        contenders = population.individuals.loc[contender_indices]
        
        if self.maximization_flag:
            return contenders[self.column].idxmax()
        else:
            return contenders[self.column].idxmin()
 
 
class StochasticUniformSelection(GeneticOperator):
    def __init__(self,
                column,
                maximization_flag):
        self.column = column
        self.maximization_flag = maximization_flag
    
    def iterate(self, population):
        N = 2*population.generation_size
        delta = 1.0/N
        offset = delta*numpy.random.rand()
        ticks = [delta*i + offset for i in xrange(N)]
        
        data = population.individuals[self.column]
        shifted_data = data - data.min()
        cdf = (shifted_data / shifted_data.sum()).cumsum()
        if not self.maximization_flag:
            cdf = 1 - cdf
        
        population.mating_pool = [numpy.argmax(cdf >= tick) for tick in ticks]
        