import numpy as np
import pandas as pd

class Population(object):
    def __init__(self, 
                 segments, 
                 phenotype=[], 
                 population_size=100, 
                 individuals=None,
                 generation_size=None, 
                 generation_counter=0, 
                 evaluation_counter=0,
                 best_log=None):
        self.segments = segments
        self.phenotype = phenotype
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

        self.best_log = best_log
            
    def randomize(self, popsize):
#         column_values = np.column_stack(tuple([np.random.randint(2 ** number_of_bits, size=popsize) for number_of_bits in self.segments.itervalues()]))
        column_values = np.column_stack(tuple([[segment.random_value() for _ in xrange(popsize)] for segment in self.segments]))
        column_names = [segment.name for segment in self.segments]
        self.individuals = pd.DataFrame(data=column_values, columns=column_names)
        self.lethal_list = range(popsize)
    
    def __str__(self):
        formatters = {}
        for segment in self.segments:
            formatters[segment.name] = segment.formatter
        
        return '\n'.join([
            'segments: ' + str(self.segments), 
            'phenotype ' + str(self.phenotype), 
            'generation_size: ' + str(self.generation_size), 
            'evaluation_counter: ' + str(self.evaluation_counter), 
            'generation_counter: ' + str(self.generation_counter), 
            'lethal_list: ' + str(self.lethal_list), 
            'survivor_list: ' + str(self.survivor_list), 
            'mating_pool: ' + str(self.mating_pool), 
            'Individuals:\n' + self.dataframe_to_string(self.individuals, formatters),
            'Best log:\n' + self.dataframe_to_string(self.best_log, formatters=formatters)])
        
    def dataframe_to_string(self, df, formatters=[]):
        if df is None:
            return 'None'
        return df.to_string(formatters=formatters)
        
    def __repr__(self):
        return str(self)
    
    def __len__(self):
        return len(self.individuals.index)