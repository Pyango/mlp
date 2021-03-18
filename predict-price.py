import logging
import multiprocessing
import os
import pickle

import pandas
import plotly.graph_objects as go
from daemonize import Daemonize

from entities.activation import all_activation_functions
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

population = Population(
    num_inputs=5,
    num_outputs=1,
    fitness_threshold=2000,  # fiat profit without fees the bot should target as good fitness
    output_activation_functions=all_activation_functions,
    initial_fitness=0,
    survival_threshold=3,  # How long networks survive before they stagnate and die
    compatibility_threshold=1,
    max_species=50,
    size=300,
    compatibility_threshold_mutate_power=.4,
    logger=logger,
)

predictions = {}

data = pandas.read_pickle(os.path.join(dir_path, 'etheur-15min-candles-2.pkl'))
# Only get one day to train
# data = data[(data.index >= "2021-03-12") & (data.index < "2021-03-13")]
# Remove unused columns
data = data.drop(['time', 'vwap', 'count'], axis=1)
# Make sure we train in the correct order
data.sort_index(inplace=True)


def compute_fitness(genome):
    genome.fitness = 0
    last_open_price = None
    for index, row in data.iterrows():
        # Define the inputs including our current account status
        genome_input = list(row)
        prediction = genome.activate(genome_input)
        price_prediction = max(-1.0, min(1.0, prediction[0]))  # clamped
        open_price = row.get('open')
        if last_open_price:
            genome.fitness += (open_price - last_open_price) * price_prediction
        last_open_price = row.get('open')
    return genome


def on_success(best):
    best_bot_outfile = open(os.path.join(dir_path, '15min-candle-predict-price-best-bot'), 'wb')
    population_outfile = open(os.path.join(dir_path, '15min-candle-predict-price-population'), 'wb')
    pickle.dump(best, best_bot_outfile)
    pickle.dump(population, population_outfile)
    best_bot_outfile.close()
    population_outfile.close()


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
    population.run(compute_fitness, on_success, on_generation=on_generation, generations=10000)


if __name__ == '__main__':
    # Set foreground to test the Daemonize
    daemon = Daemonize(app="predict-price", pid=pid, action=run, keep_fds=keep_fds, foreground=False)
    daemon.start()