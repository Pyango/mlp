import multiprocessing
import pickle

import numpy as np
from time import sleep

import krakenex
from pykrakenapi import KrakenAPI

from entities.activation import all_activation_functions
from entities.population import Population

population = Population(
    num_inputs=18,
    num_outputs=3,
    fitness_threshold=1000,  # EUR profit without fees the bot should target as good fitness
    initial_fitness=0,
    survival_threshold=3,
    compatibility_threshold=1,
    max_species=10,
    size=150,
    output_activation_functions=all_activation_functions,
)

api = krakenex.API()
k = KrakenAPI(api)
data = []
predictions = {}


def compute_fitness(genomes):
    """
    Predicts minute prices in 10 second interval
    :param genomes:
    :return:
    """
    for i in range(6):
        ticker = k.get_ticker_information("ETHEUR")
        current_price = ticker.c[0][0]
        print(f'Ticker ETHEUR Ticker: {i + 1} Current price: {current_price}')
        for genome_key, genome in genomes:
            input_array = ticker.a[0] + ticker.b[0] + ticker.c[0] + ticker.h[0] + ticker.l[0] + [ticker.o[0]] + \
                          ticker.p[0] + ticker.t[0] + ticker.v[0]
            input_array = [float(i) for i in input_array]
            pred = np.argmax(genome.activate(input_array))  # [<buy> | 0, <sell> | 1, <hold> | 2]

            if genome.key not in predictions:
                predictions[genome.key] = [{
                    'price': current_price,
                    'pred': pred,
                }]
            else:
                try:
                    prev_prediction_object = predictions[genome.key][i]
                    prev_price = prev_prediction_object.get('price')
                    prev_prediction = prev_prediction_object.get('pred')
                    if prev_prediction == 0 and current_price > prev_price:
                        # Price is higher and predicted buy signal
                        genome.fitness += abs(float(prev_price) - float(current_price))
                    elif prev_prediction == 1 and current_price < prev_price:
                        # Price is lower and predicted sell signal
                        genome.fitness += abs(float(prev_price) - float(current_price))
                    elif prev_prediction == 2:
                        genome.fitness -= 0  # we done nothing when the 3 output is be biggest
                    else:
                        # Everything else is wrong
                        genome.fitness -= abs(float(prev_price) - float(current_price))
                    predictions[genome.key][i] = {
                        'price': current_price,
                        'pred': pred,
                    }
                except IndexError:
                    predictions[genome.key].append({
                        'price': current_price,
                        'pred': pred,
                    })
        if i != 5:
            sleep(10)
        else:
            sleep(5)


def on_success(best):
    outfile = open('best-trading-bot', 'wb')
    pickle.dump(best, outfile)
    outfile.close()


def run():
    multiprocessing.freeze_support()
    # load_pickle()
    population.run(
        compute_fitness,
        on_success,
    )


if __name__ == '__main__':
    run()
