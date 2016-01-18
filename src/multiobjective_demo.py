import pandasga as pdga
import matplotlib.pyplot as plt
import numpy as np
import time

n_bits = 24

def t1(x):
    return x**2

def t2(x):
    return (x-2)**2

def z(sh, pd):
    return 1.0*sh + 2.0*pd

ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)
ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
ax3 = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=2)

X = np.linspace(-6, 6, 1001)
F1 = np.vectorize(t1)(X)
F2 = np.vectorize(t2)(X)

def plot_population(population):
    i = population.individuals
    b = population.best_log
    
    ax3.cla()
    ax3.plot(F1, F2, 'k-')
    ax3.plot(i['f1'], i['f2'], 'rx')

    ax2.cla()
    ax2.plot(b.index, b['f2'])
    
    ax1.cla()
    ax1.plot(b.index, b['f1'])
             
    plt.draw()
    time.sleep(0.05)

u = pdga.genotype.BinarySegment(name='u', number_of_bits=8)
x = pdga.genotype.BinarySegment(name='x', number_of_bits=n_bits)
y = pdga.transformations.LinearMapping(name='y', genotype_column=x, max_value=6, min_value=-6)
f1 = pdga.core.EvaluationColumn(pdga.transformations.ColumnFunction(name='f1', columns=['y'],function=t1))
f2 = pdga.transformations.ColumnFunction(name='f2', columns=['y'], function=t2)
sh = pdga.multimodal.NicheSharingCoefficient(name='sh',
                                             distance_function=pdga.multimodal.new_euclidean_distance(['y']),
                                             sharing_function=pdga.multimodal.new_linear_decay_sharing(radius=0.1))

pd = pdga.multiobjective.ParetoDomination(name='pd',
                                          comparator_mapping={
                                            'f1': pdga.multiobjective.minimize_comparator,
                                            'f2': pdga.multiobjective.minimize_comparator})

z = pdga.transformations.ColumnFunction(name='z',
                                        columns=['sh', 'pd'],
                                        function=z,
                                        normalize_columns=True)

sl = pdga.selection.SelectWorstNLethals(column='z', maximize=False)
sm = pdga.selection.StochasticUniformSelection(column='z', maximize=False)
c = pdga.operators.Crossover(crossover_probability=1.0)
m = pdga.operators.Mutation(mutation_probability=0.05)
d = pdga.operators.DecodeAndEvaluate()
l = pdga.operators.LogBest(column='z', maximize=False)
p = pdga.operators.PeriodicOperator(generation_frequency=1,
                                    iteration_callback=plot_population)

pop = pdga.population.Population(
          segments=[u, x],
          phenotype=[y, f1, f2, sh, pd, z],
          population_size=100,
          generation_size=20)
 
ga = pdga.core.Scheduler(name='test',
                population=pop,
                operators=[sl, sm, c, m, d, l, p])
 
plt.ion()
plt.show()

ga.run_ga(1000)
print pop
