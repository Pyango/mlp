from entities.population import Population, xor2

population = Population(
    num_inputs=2,
    num_outputs=1,
    fitness_threshold=3.99,
    initial_fitness=4.0,
    survival_threshold=3,
    compatibility_threshold=1,
    max_species=10,
    size=150,
)


def compute_fitness(genomes):
    for genome_key, genome in genomes:
        genome.fitness = 4.0
        for i in xor2:
            result = genome.activate(i[:2])
            genome.fitness -= (result[0] - i[-1]) ** 2


def on_success(best):
    for i in xor2:
        print(f'{i} -> {best.activate(i[:2])}')


population.run(compute_fitness)
