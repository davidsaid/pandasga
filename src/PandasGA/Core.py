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
    
class GeneticOperator(GABaseObject):
    def initialize(self, population):
        pass

    def iterate(self, population):
        pass
    
    def finalize(self, population):
        pass

class TransformationColumn(GABaseObject):
    def __init__(self,name, function, formatter=str):
        self.function = function
        self.name = name
        self.formatter = formatter
    
    def evaluate(self, population):
        population.individuals[self.name] = \
            self.function(population)

class EvaluationColumn(TransformationColumn):
    def __init__(self,
                 column):
        self.column = column
    
    def evaluate(self, population):
        self.column.evaluate(population)
        population.evaluation_counter += len(population)

class Scheduler(GABaseObject):
    def __init__(self, \
                 population, \
                 name, \
                 operators):
        self.population = population
        self.name = name
        self.operators = operators
        
    def __str__(self):
        return '\n'.join(\
                 ['name: ' + str(self.name), \
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
    
    def run_ga(self, n):
        self.initialize()
        for _ in xrange(n):
            self.iterate()
            self.population.generation_counter += 1
        self.finalize()