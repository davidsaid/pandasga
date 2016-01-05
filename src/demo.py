from PandasGA.Core import OptimizationObjective, BinarySegment, SelectLethals, KTournamentSelection, Crossover, Mutate, Evaluation, Population, Scheduler


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
    
    pop = Population(\
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