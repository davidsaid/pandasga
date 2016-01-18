from core import TransformationColumn
import numpy.matlib

def maximize_comparator(a, b):
    return 1.0 if max(a, b) == a else 0.0

def minimize_comparator(a, b):
    return 1.0 if min(a, b) == a else 0.0

class ParetoDomination(TransformationColumn):
    def __init__(self, name, comparator_mapping):
        ''' comparator_mapping is a map of column name to a function, 
        that given two values returns 1 if the first is better or equal 
        than the second and 0 otherwise. This library has a collectio of 
        such functions'''
        self.name = name
        self.comparator_mapping = {}
        for columm, scalar_comparator in comparator_mapping.iteritems():
            self.comparator_mapping[columm] = numpy.vectorize(scalar_comparator)
    
    def evaluate(self, population):
        population.individuals[self.name] = \
            numpy.sum(self.domination_count(population), axis=1)
    
    def domination_count(self, population):
        N = len(population)
        D = numpy.ones((N, N))
        for column, comparator_fn in self.comparator_mapping.iteritems():
            I = numpy.matlib.repmat(population.individuals[column], N, 1)
            J = numpy.transpose(I)
            D *= comparator_fn(I, J)
        return D