import logging
import multiprocessing
import os
import pickle
import random

import pandas

from entities.activation import all_activation_functions, clamped_activation
from entities.population import Population

dir_path = os.path.dirname(os.path.realpath(__file__))
pid = os.path.join(dir_path, 'predict-price.pid')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False
fh = logging.FileHandler(os.path.join(dir_path, 'predict-price.log'), 'w')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
keep_fds = [fh.stream.fileno()]

data = pandas.read_pickle(os.path.join(dir_path, 'etheur-15min-candles-2.pkl'))
# Only get one day to train
# Remove unused columns
# Make sure we train in the correct order
data.sort_index(inplace=True)
data['last_open'] = data['open'].shift(1)
data['result'] = data.apply(lambda row: 1 if row.open > row.last_open else -1 if row.open < row.last_open else 0, axis=1)
data = data.drop(['time', 'vwap', 'count', 'last_open'], axis=1)


def compute_fitness(genome, population):
    genome.fitness = 0
    for index, row in population.data.iterrows():
        # Define the inputs including our current account status
        genome_input = list(row[:-1])
        prediction = genome.activate(genome_input)
        price_prediction = max(-1.0, min(1.0, prediction[0]))  # clamped
        genome.fitness += row.result * price_prediction
    return genome


def on_success(best, population):
    logger.debug('Solution found!!!!!!!!!!!!!\n\n')
    logger.debug(f'Best genome predictions:')
    best_bot_outfile = open(os.path.join(dir_path, '15min-candle-predict-price-best-bot'), 'wb')
    population_outfile = open(os.path.join(dir_path, '15min-candle-predict-price-population'), 'wb')
    pickle.dump(best, best_bot_outfile)
    pickle.dump(population, population_outfile)
    best_bot_outfile.close()
    population_outfile.close()

    for index, row in population.data.iterrows():
        # Define the inputs including our current account status
        genome_input = list(row[:-1])
        prediction = best.activate(genome_input)
        price_prediction = max(-1.0, min(1.0, prediction[0]))  # clamped
        logger.debug(f'{genome_input} -> {price_prediction}, actual {row.result}')


def on_generation(best, population):
    most_complex = None
    for genome in population.genomes.values():
        if not most_complex:
            most_complex = genome
        else:
            if sum(genome.complexity) > sum(most_complex.complexity):
                most_complex = genome
                logger.debug(
                    f'Most complex genome is {genome.key} with {genome.complexity} and age {genome.generation}')

    best_bot_outfile = open(os.path.join(dir_path, '15min-candle-predict-price-best-bot'), 'wb')
    pickle.dump(best, best_bot_outfile)
    best_bot_outfile.close()


def run():
    multiprocessing.freeze_support()

    for i in range(2, 10):
        logger.debug(f'Running trial {i-2}...')
        nrows = range(data.shape[0])
        ix = random.randint(nrows.start, nrows.stop - i)
        sample = data.iloc[ix:ix + i, :]
        import pdb;pdb.set_trace()
        try:
            population = pandas.read_pickle(os.path.join(dir_path, '15min-candle-predict-price-population'))
            population.data = sample
            population.fitness_threshold = i - 0.01
        except FileNotFoundError:
            logger.debug('Population not found! Starting from scratch!')
            population = Population(
                num_inputs=5,
                num_outputs=1,
                fitness_threshold=i - 0.01,  # 80% correct
                output_activation_functions=all_activation_functions,
                output_activation_function=clamped_activation,
                initial_fitness=0,
                survival_threshold=3,  # How long networks survive before they stagnate and die
                compatibility_threshold=1,
                max_species=10,
                size=50,
                compatibility_threshold_mutate_power=4,
                logger=logger,
                data=sample,
            )
        population.run(compute_fitness, on_success, on_generation=on_generation, generations=10000)


if __name__ == '__main__':
    run()
