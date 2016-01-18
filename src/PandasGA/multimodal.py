from core import TransformationColumn
import numpy.matlib

def euclidean_distance(population, columns):
    d_square = 0
    N = len(population)
    
    for column in columns:
        x = numpy.matlib.repmat(population.individuals[column], N, 1)
        y = numpy.transpose(x)
        d_square += numpy.square(x - y)
    return numpy.sqrt(d_square)

def new_euclidean_distance(columns):
    return lambda p : euclidean_distance(p, columns)

def linear_decay_sharing(distance, r):
    o = 0.0
    d = abs(distance)
    if r > 0 and d <= r:
        o = 1.0 - d/r
    return o

def new_linear_decay_sharing(radius):
    return lambda distance: linear_decay_sharing(distance, radius)

class NicheSharingCoefficient(TransformationColumn):
    def __init__(self, name, distance_function, sharing_function):
        self.name = name
        self.distance_function = distance_function
        self.sharing_function = numpy.vectorize(sharing_function)
    
    def evaluate(self, population):
        d = self.distance_function(population)
        population.individuals[self.name] = \
            numpy.sum(self.sharing_function(d), axis=1)