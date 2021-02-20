from entities.activation import sigmoid_activation
from entities.population import Population

xor2 = [
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
]

xor3 = [
    [0, 0, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 1],
]

xand = [
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1],
]

nand = [
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1],
]

population = Population(
    num_inputs=2,
    num_outputs=1,
    fitness_threshold=3.99,
    initial_fitness=4.0,
    survival_threshold=0,
    compatibility_threshold=1,
    max_species=20,
    size=300,
    output_activation_function=sigmoid_activation,
    compatibility_threshold_mutate_power=.1,
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


population.run(compute_fitness, on_success, generations=3000)
