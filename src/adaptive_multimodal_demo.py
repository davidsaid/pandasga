import pandasga as pdga
import matplotlib.pyplot as plt
import numpy as np
import math

n_bits = 24

def function_z(y, t):
    return np.exp(-3 * y) * np.sin(12 * math.pi * y - np.pi * t/50) 

Y = np.linspace(0, 1, 1001)


ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)
ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
ax3 = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=2)

def plot_log(population, logger):
    Z = function_z(Y, population.generation_counter)
    ax1.cla()
    ax1.plot(Y, Z, 'b-')
    ax1.plot(population.individuals['y'], population.individuals['z'], 'rx')
    ax1.set_title('generation %d' % population.generation_counter)
    ax1.set_xlabel('y')
    ax1.set_ylabel('z')

    ax2.cla()
    ax2.plot(Y, Z, 'b-')
    ax2.plot(population.individuals['y'], population.individuals['z'], 'rx')
    ax2.set_title('generation %d' % population.generation_counter)
    ax2.set_xlabel('y')
    ax2.set_ylabel('z')
    
    best = population.individuals['z'].idxmax()
    best_x = population.individuals.loc[best, 'y']
    best_z = population.individuals.loc[best, 'z']
    window_size = 0.05
    ax2.set_xlim([best_x - 0.5 * window_size, best_x + 0.5 * window_size])
    ax2.set_ylim([best_z - 0.5 * window_size, best_z + 0.5 * window_size])
    
    ax3.plot(logger.best_log['z'], 'k-')
    ax3.set_title('best found so far')
    ax3.set_xlabel('evaluations')
    ax3.set_ylabel('z')
         
    plt.draw()

u = pdga.genotype.BinarySegment(name='u', number_of_bits=8)
x = pdga.genotype.BinarySegment(name='x', number_of_bits=n_bits)
y = pdga.transformations.LinearMapping(name='y', genotype_column=x)     
z = pdga.core.EvaluationColumn(
        pdga.core.TransformationColumn(name='z',
                                       function=lambda p : function_z(p.individuals['y'], p.generation_counter)))
sh = pdga.multimodal.NicheSharingCoefficient(name='sh',
                                             distance_function=pdga.multimodal.new_euclidean_distance(['y']),
                                             sharing_function=pdga.multimodal.new_linear_decay_sharing(radius=0.1))

z_sh = pdga.transformations.ColumnFunction(name='z_sh',
                                         columns=['z', 'sh'],
                                         function = lambda z, sh : z/sh)

sl = pdga.selection.SelectWorstNLethals(column='z_sh')
sm = pdga.selection.StochasticUniformSelection(column='z_sh')
c = pdga.operators.Crossover(crossover_probability=1.0)
m = pdga.operators.Mutation(mutation_probability=0.85)
d = pdga.operators.DecodeAndEvaluate()
l = pdga.operators.LogBest(column='z')
p = pdga.operators.PeriodicOperator(generation_frequency=1,
                                    iteration_callback=lambda p: plot_log(p, l))

pop = pdga.population.Population(
          segments=[u, x],
          phenotype=[y, z, sh, z_sh],
          population_size=200,
          generation_size=150)
 
ga = pdga.core.Scheduler(name='test',
                population=pop,
                operators=[sl, sm, c, m, d, l, p])
 
plt.ion()
plt.show()

ga.run_ga(150)
print pop
