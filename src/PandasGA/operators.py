from core import GeneticOperator

import numpy as np
import pandas as pd

class DecodeAndEvaluate(GeneticOperator):
    def apply(self, population):
        for column in population.phenotype:
            column.evaluate(population)
    
    def iterate(self, population):
        self.apply(population)
    
    def initialize(self, population):
        self.apply(population)

class Crossover(GeneticOperator):
    def __init__(self,
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
    def __init__(self,
                 mutation_probability):
        self.mutation_probability = mutation_probability
    
    def iterate(self, population):
        individuals = population.individuals
        for i, row in individuals.loc[population.lethal_list].iterrows():
            for segment in population.segments:
                if np.random.rand() <= self.mutation_probability:
                    individuals.set_value(i, segment.name, segment.mutate(row[segment.name]))
                    
class PeriodicOperator(GeneticOperator):
    def __init__(self,
                 generation_frequency=1,
                 iteration_callback=None,
                 evaluation_frequency=None,
                 evaluation_callback=None):
        self.generation_frequency = generation_frequency
        self.iteration_callback = iteration_callback
        self.evaluation_frequency = evaluation_frequency
        self.evaluation_callback = evaluation_callback
    
    def should_trigger(self, counter, frequency):
        if frequency:
            return counter % frequency == 0
        return False
    
    def iterate(self, population):
        if self.should_trigger(population.generation_counter, self.generation_frequency) \
                and self.iteration_callback is not None:
            self.iteration_callback(population)
        if self.should_trigger(population.evaluation_counter, self.evaluation_frequency) \
                and self.evaluation_callback is not None:
            self.evaluation_callback(population)

class LogBest(PeriodicOperator):
    def __init__(self,
        column,
        maximize=True,
        generation_frequency=None,
        iteration_callback=None,
        evaluation_frequency=1,
        evaluation_callback=None,
        plot_callback=None):
        PeriodicOperator.__init__(self,
                                  generation_frequency,
                                  iteration_callback,
                                  evaluation_frequency,
                                  evaluation_callback)
        self.column = column
        self.maximize = maximize
        ul = lambda pop: self.update_log(pop)
        self.iteration_callback = ul
        self.evaluation_callback = ul
        self.best_log = None
        self.plot_callback = plot_callback
        
    def update_log(self, population):
        best_index = self.get_best_index(population)
        best = population.individuals.loc[best_index]
        if self.best_log is None:
            self.best_log = pd.DataFrame(columns=population.individuals.columns)
        self.best_log.loc[population.evaluation_counter] = best
        if self.plot_callback:
            self.plot_callback(self.best_log)

    def get_best_index(self, population):
        if self.maximize:
            return population.individuals[self.column].idxmax()
        else:
            return population.individuals[self.column].idxmin()