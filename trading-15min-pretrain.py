import pickle

import krakenex
import numpy as np
import pandas
from pykrakenapi import KrakenAPI

from entities.activation import all_activation_functions
from entities.population import Population

population = Population(
    num_inputs=9,
    num_outputs=4,
    fitness_threshold=2000,  # fiat profit without fees the bot should target as good fitness
    initial_fitness=0,
    survival_threshold=3,
    compatibility_threshold=1,
    max_species=20,
    size=300,
    output_activation_functions=all_activation_functions,
    compatibility_threshold_mutate_power=.8,
)

api = krakenex.API()
k = KrakenAPI(api)
predictions = {}

data = pandas.read_pickle("./etheur-15min-candles.pkl")

trade_size_fiat = 100


def compute_fitness(genomes):
    for genome_key, genome in genomes:
        genome.fitness = 0
        fiat_account = 1000
        crypto_account = 0
        close_price = 0
        for index, row in data.iterrows():
            # Define the inputs including our current account status
            genome_input = list(row)[1:] + [fiat_account, crypto_account]
            prediction = genome.activate(genome_input)
            trading_decision = np.argmax(prediction[:2])
            # TODO: Simpler method?
            if prediction[3] >= 1:
                trade_size_percentage = 1
            elif prediction[3] <= 0:
                trade_size_percentage = 0
            else:
                trade_size_percentage = prediction[3]
            fiat_trade_amount = trade_size_fiat * trade_size_percentage

            open_price = row.get('open')
            close_price = row.get('close')
            if trading_decision == 0:
                # We want to buy
                if fiat_account <= 0:
                    # print('We dont have any fiat money left!')
                    continue
                # Make a trade for the open price
                if not fiat_trade_amount:
                    # print('We cant make a trade for nothing!')
                    continue

                # In case we want to sell more fiat than we have
                if fiat_account <= fiat_trade_amount:
                    fiat_trade_amount = fiat_account

                amount_crypto_to_buy = fiat_trade_amount / open_price

                crypto_account += amount_crypto_to_buy
                fiat_account -= fiat_trade_amount

            elif trading_decision == 1:
                # We want to sell
                if crypto_account <= 0:
                    # print('We dont have any crypto left!')
                    continue
                # Make a trade for the open price
                if not fiat_trade_amount:
                    # print('We cant make a trade for nothing!')
                    continue

                amount_crypto_to_sell = fiat_trade_amount / open_price
                # In case we want to sell more crypto than we have
                if crypto_account <= amount_crypto_to_sell:
                    amount_crypto_to_sell = crypto_account

                crypto_account -= amount_crypto_to_sell
                fiat_account += amount_crypto_to_sell * open_price
            elif trading_decision == 2:
                # print('Hold your horses we hang tight till the next trade!')
                continue
        # Sell all open crypto for the last closing price
        if crypto_account > 0:
            fiat_account += crypto_account * close_price
            crypto_account = 0
        genome.fitness = fiat_account


def on_success(best):
    outfile = open('best-bot-15min-candle-trading', 'wb')
    pickle.dump(best, outfile)
    outfile.close()


population.run(compute_fitness, on_success, generations=1000)
