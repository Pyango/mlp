import logging
import multiprocessing
import os
import pickle

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

dir_path = os.path.dirname(os.path.realpath(__file__))
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False
fh = logging.FileHandler(os.path.join(dir_path, 'xor2.log'), 'w')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

population = Population(
    num_inputs=2,
    num_outputs=1,
    fitness_threshold=3.99,
    initial_fitness=4.0,
    survival_threshold=0,
    compatibility_threshold=1,
    max_species=20,
    size=150,
    output_activation_functions=[sigmoid_activation],
    compatibility_threshold_mutate_power=.1,
    logger=logger,
)


def compute_fitness(genome):
    genome.fitness = 4.0
    for i in xor2:
        result = genome.activate(i[:2])
        genome.fitness -= (result[0] - i[-1]) ** 2
    return genome


def on_success(best):
    for i in xor2:
        print(f'{i} -> {best.activate(i[:2])}')
    outfile = open('best-xor2', 'wb')
    pickle.dump(best, outfile)
    outfile.close()


def load_pickle():
    infile = open('best-xor2', 'rb')
    best = pickle.load(infile)
    infile.close()
    for i in xor2:
        print(f'{i} -> {best.activate(i[:2])}')


def run():
    multiprocessing.freeze_support()
    # load_pickle()
    population.run(compute_fitness, on_success, generations=3000)


if __name__ == '__main__':
    run()
