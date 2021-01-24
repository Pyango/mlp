from entities.population import Population, xor2

population = Population(
    num_inputs=2,
    num_outputs=1,
    fitness_threshold=3.99,
    initial_fitness=4.0,
)


def compute_fitness(genomes):
    for genome_key, genome in genomes:
        genome.fitness = 4.0
        for i in xor2:
            result = genome.activate(i[:2])
            genome.fitness -= abs(result[0] - i[-1])


population.run(compute_fitness)
