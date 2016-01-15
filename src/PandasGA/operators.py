from core import GeneticOperator

import numpy as np
import pandas as pd

class Transformation(GeneticOperator):
    def get_column_descriptors(self, population):
        raise NotImplemented('This is an abstract method that needs to be implemented by sub-classes')
    
    def apply(self, population):
        individuals = population.individuals
        for column in self.get_column_descriptors(population):
            individuals.loc[population.lethal_list, column.name] = \
                column.function(individuals.loc[population.lethal_list])
        population.evaluation_counter += len(population.lethal_list)
    
    def iterate(self, population):
        self.apply(population)
    
    def initialize(self, population):
        self.apply(population)

class Evaluation(Transformation):
    def get_column_descriptors(self, population):
        return population.targets

class Decode(Transformation):
    def get_column_descriptors(self, population):
        return population.phenotype

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
    
class Mutation(GeneticOperator):
    def __init__(self, \
                 mutation_probability):
        self.mutation_probability = mutation_probability
    
    def iterate(self, population):
        individuals = population.individuals
        for i, row in individuals.loc[population.lethal_list].iterrows():
            for segment in population.segments:
                if np.random.rand() <= self.mutation_probability:
                    individuals.set_value(i, segment.name, segment.mutate(row[segment.name]))
                    
class PeriodicOperator(GeneticOperator):
    def __init__(self,\
                 generation_frequency=1,\
                 iteration_callback=None,\
                 evaluation_frequency=None,\
                 evaluation_callback=None):
        self.generation_frequency = generation_frequency
        self.iteration_callback = iteration_callback
        self.evaluation_frequency = evaluation_frequency
        self.evaluation_callback=evaluation_callback
    
    def should_trigger(self, counter, frequency):
        if frequency:
            return counter % frequency == 0
        return False
    
    def iterate(self, population):
        if self.should_trigger(population.generation_counter, self.generation_frequency)\
                and self.iteration_callback is not None:
            self.iteration_callback(population)
        if self.should_trigger(population.evaluation_counter, self.evaluation_frequency)\
                and self.evaluation_callback is not None:
            self.evaluation_callback(population)

class LogBest(PeriodicOperator):
    def __init__(self, 
        generation_frequency=None, 
        iteration_callback=None, 
        evaluation_frequency=1, 
        evaluation_callback=None):
        PeriodicOperator.__init__(self, generation_frequency=generation_frequency, iteration_callback=iteration_callback, evaluation_frequency=evaluation_frequency, evaluation_callback=evaluation_callback)
        ul = lambda pop: self.update_log(pop)
        self.iteration_callback = ul
        self.evaluation_callback = ul
        
    def update_log(self, population):
        best_index = population.survivor_list[-1]
        best = population.individuals.loc[best_index]
        if population.best_log is None:
            population.best_log = pd.DataFrame(columns=population.individuals.columns)
        
        population.best_log.loc[population.evaluation_counter] = best
