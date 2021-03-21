import logging
import multiprocessing
import os
import pickle

from entities.activation import all_activation_functions, relu_activation, sigmoid_activation
from entities.population import Population

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

dir_path = os.path.dirname(os.path.realpath(__file__))
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False
fh = logging.FileHandler(os.path.join(dir_path, 'xor3.log'), 'w')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

population = Population(
    num_inputs=3,
    num_outputs=1,
    fitness_threshold=7.99,
    initial_fitness=8.0,
    survival_threshold=0,
    compatibility_threshold=1,
    max_species=20,
    size=10,
    output_activation_functions=[relu_activation, relu_activation],
    compatibility_threshold_mutate_power=.1,
    logger=logger,
)


def compute_fitness(genome):
    genome.fitness = 8.0
    for i in xor3:
        result = genome.activate(i[:3])
        genome.fitness -= (result[0] - i[-1]) ** 2
    return genome


def on_generation(best, population):
    if not population.generation % 100:
        print('')
        print('')
        print('')
        print(f'Generation {population.generation}')
        for i in xor3:
            print(f'{i} -> {best.activate(i[:3])}')
        print('')
        print('')
        print('')


def on_success(best):
    for i in xor3:
        print(f'{i} -> {best.activate(i[:3])}')
    outfile = open('best-xor3', 'wb')
    pickle.dump(best, outfile)
    outfile.close()


def load_pickle():
    infile = open('best-xor3', 'rb')
    best = pickle.load(infile)
    infile.close()
    for i in xor3:
        print(f'{i} -> {best.activate(i[:3])}')


def run():
    multiprocessing.freeze_support()
    population.run(compute_fitness, on_success, on_generation=on_generation, generations=3000)


if __name__ == '__main__':
    run()
