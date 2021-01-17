from entities.population import Population

xor = [
    [0, 0, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 1],
]

population = Population(
    num_inputs=3,
    num_outputs=1,
    fitness_threshold=3.99,
    initial_fitness=4.0,
)


def compute_fitness(genomes):
    for genome_key, genome in genomes:
        for i in xor:
            result = genome.activate(i[:3])
            genome.fitness -= (result[0] - i[-1]) ** 2
        genome.mutate()


population.run(compute_fitness)
