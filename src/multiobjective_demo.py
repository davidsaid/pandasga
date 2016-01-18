import pandasga as pdga
import matplotlib.pyplot as plt
import numpy as np
import math
import time

n_bits = 24
y_min = -7*np.pi
y_max = 2*np.pi
t1 = lambda x: math.sin(x)
t2 = lambda x: math.cos(x)

def z(sh, pd):
    return 1.0*sh + 2.0*pd

ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)
ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
ax3 = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=2)

X = np.linspace(y_min, y_max, 1001)
F1 = np.vectorize(t1)(X)
F2 = np.vectorize(t2)(X)
        
def new_plot_log(logger, ax):
    return lambda p: plot_log(p, logger, ax)

def plot_log(population):
    i = population.individuals
    
    ax1.cla()
    ax1.plot(F1, F2, 'k-')
    ax1.plot(i['f1'], i['f2'], 'rx')
    ax1.set_xlabel('f1')
    ax1.set_xlim([-1.1, 1.1])
    ax1.set_ylabel('f2')
    ax1.set_ylim([-1.1, 1.1])
    
    ax2.cla()
    ax2.plot(F1, F2, 'k-')
    ax2.plot(i['f1'], i['f2'], 'rx')
    ax2.set_xlabel('f1')
    ax2.set_xlim([-1.1, 0.1])
    ax2.set_ylabel('f2')
    ax2.set_ylim([-1.1, 0.1])
        
    ax3.cla()
    ax3.plot(X, F1, 'g-')
    ax3.plot(X, F2, 'b-')
    ax3.plot(i['y'], i['f1'], 'go')
    ax3.plot(i['y'], i['f2'], 'bo')
    ax3.set_xlim([y_min, y_max])
    ax3.set_ylim([-1.1, 1.1])
             
    plt.draw()
    time.sleep(0.05)

u = pdga.genotype.BinarySegment(name='u', number_of_bits=8)
x = pdga.genotype.BinarySegment(name='x', number_of_bits=n_bits)
y = pdga.transformations.LinearMapping(name='y',
                                       genotype_column=x,
                                       min_value=y_min,
                                       max_value=y_max)
f1 = pdga.core.EvaluationColumn(pdga.transformations.ColumnFunction(name='f1', columns=['y'],function=t1))
f2 = pdga.transformations.ColumnFunction(name='f2', columns=['y'], function=t2)
sh = pdga.multimodal.NicheSharingCoefficient(name='sh',
                                             distance_function=pdga.multimodal.new_euclidean_distance(['y']),
                                             sharing_function=pdga.multimodal.new_linear_decay_sharing(radius=0.1))

pd = pdga.multiobjective.ParetoDomination(name='pd',
                                          comparator_mapping={
                                            'f1': pdga.multiobjective.maximize_comparator,
                                            'f2': pdga.multiobjective.maximize_comparator})

z = pdga.transformations.ColumnFunction(name='z',
                                        columns=['sh', 'pd'],
                                        function=z,
                                        normalize_columns=True)

sl = pdga.selection.SelectWorstNLethals(column='z', maximize=False)
sm = pdga.selection.StochasticUniformSelection(column='z', maximize=False)
c = pdga.operators.Crossover(crossover_probability=1.0)
m = pdga.operators.Mutation(mutation_probability=0.5)
d = pdga.operators.DecodeAndEvaluate()
p = pdga.operators.PeriodicOperator(generation_frequency=1,
                                    iteration_callback=plot_log)

pop = pdga.population.Population(
          segments=[u, x],
          phenotype=[y, f1, f2, sh, pd, z],
          population_size=100,
          generation_size=5)
 
ga = pdga.core.Scheduler(name='test',
                population=pop,
                operators=[sl, sm, c, m, d, p])
 
plt.ion()
plt.show()

ga.run_ga(1000)
print pop
