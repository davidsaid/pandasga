'''
Created on Jan 1, 2016

@author: dsaid
'''
import numpy as np
import pandas as pd
from webbrowser import Chrome

class GABaseObject(object):
    def __init__(self, **kwargs):
        for param, val in kwargs.items():
            self.__setattr__(param, val)    

    def to_string(self, delim):
        return '%(class)s(%(params)s)' % \
            { 'class':type(self).__name__, \
              'params':delim.join([str(prop) + ' = ' + repr(value) for prop, value in vars(self).iteritems()]) }
            
    def __repr__(self):
        return self.to_string(', ')

    def __str__(self):
        return repr(self)

class BinarySegmentPopulation(GABaseObject):
    def __init__(self, \
                 segments, \
                 targets, \
                 population_size=100, \
                 individuals=None,
                 generation_size=None, \
                 generation_counter=0, \
                 evaluation_counter=0):
        self.segments = segments
        self.targets = targets
        self.generation_counter = generation_counter
        self.evaluation_counter = evaluation_counter
        if not generation_size:
            self.generation_size = population_size
        else:
            self.generation_size = generation_size
        self.lethal_list = None
        self.survivor_list = None
        self.mating_pool = None
        
        if individuals:
            self.individuals = individuals
        else:
            self.individuals = None
            self.randomize(population_size) 

    def randomize(self, popsize):
#         column_values = np.column_stack(tuple([np.random.randint(2 ** number_of_bits, size=popsize) for number_of_bits in self.segments.itervalues()]))
        column_values = np.column_stack(tuple([[segment.random_value() for _ in xrange(popsize)] for segment in self.segments]))
        column_names = [segment.name for segment in self.segments]
        self.individuals = pd.DataFrame(data=column_values, columns=column_names)
        self.lethal_list = range(popsize)
    
    def __str__(self):
        segment_formatters = {}
        for segment in self.segments:
            segment_formatters[segment.name] = segment.formatter
        
        return '\n'.join([\
            'segments: ' + str(self.segments), \
            'targets: ' + str(self.targets), \
            'generation_size: ' + str(self.generation_size), \
            'evaluation_counter: ' + str(self.evaluation_counter), \
            'generation_counter: ' + str(self.generation_counter), \
            'lethal_list: ' + str(self.lethal_list), \
            'survivor_list: ' + str(self.survivor_list), \
            'mating_pool: ' + str(self.mating_pool), \
            'Individuals:\n' + self.individuals.to_string(formatters=segment_formatters)])
                
    def __repr__(self):
        return str(self)
    
    def __len__(self):
        return len(self.individuals.index)
    
class GeneticOperator(GABaseObject):
    def initialize(self, population):
        pass

    def iterate(self, population):
        pass
    
    def finalize(self, population):
        pass

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

class Crossover(GeneticOperator):
    def __init__(self, \
                 crossover_probability):
        self.crossover_probability = crossover_probability
    
    def iterate(self, population):
        males = population.mating_pool[:population.generation_size]
        females = population.mating_pool[population.generation_size:]

        segment_names = [segment.name for segment in population.segments]
        offspring = pd.DataFrame(columns=segment_names, index=population.lethal_list)
        for j in xrange(population.generation_size):
            for segment in population.segments:
                a = population.individuals.loc[males[j], segment.name]
                b = population.individuals.loc[females[j], segment.name]
                if np.random.rand() <= self.crossover_probability:
                    c = segment.crossover(a, b)
                else:
                    c = a
                offspring.loc[population.lethal_list[j], segment.name] = c
        
        population.individuals.loc[population.lethal_list] = offspring
    
class Mutate(GeneticOperator):
    def __init__(self, \
                 mutation_probability):
        self.mutation_probability = mutation_probability
    
    def iterate(self, population):
        individuals = population.individuals
        for i, row in individuals.loc[population.lethal_list].iterrows():
            for segment in population.segments:
                if np.random.rand() <= self.mutation_probability:
                    individuals.set_value(i, segment.name, segment.mutate(row[segment.name]))

class Evaluation(GeneticOperator):
    def evaluate(self, population):
        individuals = population.individuals
        for target in population.targets:
            individuals.loc[population.lethal_list, target.name] = target.function(individuals.loc[population.lethal_list])
        population.evaluation_counter += len(population.lethal_list)
    
    def iterate(self, population):
        self.evaluate(population)
    
    def initialize(self, population):
        self.evaluate(population)
    
    def finalize(self, population):
        pass
    
class SelectLethals(GeneticOperator):
    def __init__(self, axis=0):
        self.axis=0
    
    def iterate(self, population):
        target = population.targets[self.axis]
        index = pop.individuals.sort_values(by=target.name, ascending=target.maximization_flag).index
        population.lethal_list = index[:population.generation_size]
        population.survivor_list = index[population.generation_size:]
    
    def initialize(self, population):
        pass
    
    def finalize(self, population):
        pass

class BasePeriodicOperator(GeneticOperator):
    def __init__(self, \
                 generation_counter=0, \
                 evaluation_counter=0, \
                 generation_frequency=None, \
                 evaluation_frequency=None, \
                 **kwargs):
        self.generation_counter = generation_counter
        self.evaluation_counter = evaluation_counter
        self.generation_frequency = generation_frequency
        self.evaluation_frequency = evaluation_frequency
        super(BasePeriodicOperator, self).__init__(**kwargs)
    
    def iteration_callback(self, population):
        pass
    
    def evaluation_callback(self, population):
        pass
    
    def should_trigger(self, counter, frequency):
        if frequency:
            return counter % frequency == 0
        return False
    
    def iterate(self, population):
        if self.should_trigger(population.generation_counter, self.generation_frequency):
            self.iteration_callback(population)
        if self.should_trigger(population.evaluation_counter, self.evaluation_frequency):
            self.evaluation_callback(population)

class Scheduler(GABaseObject):
    def __init__(self, \
                 population, \
                 name, \
                 operators):
        self.population = population
        self.name = name
        self.operators = operators
        
    def __str__(self):
        return '\n'.join([\
                          'name: ' + str(self.name), \
                          'operators: \n' + '\n'.join(['\t' + str(o) for o in self.operators]), \
                          'population: \n' + str(self.population)])

    def initialize(self):
        for o in self.operators:
            o.initialize(self.population)

    def iterate(self):
        for o in self.operators:
            o.iterate(self.population)
    
    def finalize(self):
        for o in self.operators:
            o.finalize(self.population)
    
    def runGA(self, n):
        self.initialize()
        for _ in xrange(n):
            self.iterate()
            self.population.generation_counter += 1
        self.finalize()

class OptimizationObjective(GABaseObject):
    def __init__(self, name, function, maximization_flag, formatter=str):
        self.name = name
        self.function = function
        self.maximization_flag = maximization_flag
        self.formatter = formatter

class BinarySegment(GABaseObject):
    def __init__(self, name, number_of_bits, formatter=None):
        self.name = name
        self.number_of_bits = number_of_bits
        if formatter is None:
            formatter = lambda x : self.to_binary(x)
        self.formatter = formatter
        
    def to_binary(self, x):
        return '0b' + np.binary_repr(x, self.number_of_bits)
    
    def crossover(self, a, b):
        cross_point = np.random.randint(self.number_of_bits + 1)
        return self.head(a, cross_point) + self.tail(b, cross_point)
    
    def mutate(self, c):
        return int(c) ^ (1<<np.random.randint(self.number_of_bits))
    
    def tail(self, c, cross_point):
        return int(c) & cross_point
    
    def head(self, c, cross_point):
        return int(c) - self.tail(c, cross_point)
    
    def random_value(self):
        return np.random.randint(2**self.number_of_bits)

if __name__ == '__main__':
    def number_of_ones(num):
        return sum([1 if bit == '1' else 0 for bit in bin(num)])
    
    z = OptimizationObjective(name='z', \
                              function=lambda (row): row['y'].apply(number_of_ones), \
                              maximization_flag=True)
    
    u = BinarySegment(name='u', number_of_bits=3)
    y = BinarySegment(name='y', number_of_bits=8)
    
    sl = SelectLethals()
    sm = KTournamentSelection(tournament_size=2)
    x = Crossover(crossover_probability=1.0)
    m = Mutate(mutation_probability=0.5)
    e = Evaluation()
    
    pop = BinarySegmentPopulation(\
              segments=[u, y], \
              targets=[z],
              population_size=100,
              generation_size=10)
     
    ga = Scheduler(name='test', \
                    population=pop, \
                    operators=[sl, sm, x, m, e])
     
#     ind = pop.individuals
#     ind.loc[::2, 'needs_update'] = False
     
    ga.runGA(50)
    print pop
