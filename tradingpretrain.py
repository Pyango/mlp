import logging
import multiprocessing
import pickle

import numpy as np
import pandas
import plotly.graph_objects as go
from daemonize import Daemonize

from entities.activation import all_activation_functions
from entities.population import Population

population = Population(
    num_inputs=10,
    num_outputs=4,
    fitness_threshold=2000,  # fiat profit without fees the bot should target as good fitness
    output_activation_functions=all_activation_functions,
    initial_fitness=0,
    survival_threshold=10,  # How long networks survive before they stagnate and die
    compatibility_threshold=1,
    max_species=20,
    size=150,
    compatibility_threshold_mutate_power=.4,
)

predictions = {}

data = pandas.read_pickle("./etheur-15min-candles.pkl")
# Make sure we train in the correct order
data.sort_index(inplace=True)
trade_size_fiat = 100
# We prefer to pay the fees in fiat. If we cant we pay them in crypto
trading_fee_percent = 0.0016


def compute_fitness(genome):
    genome.fitness = 0
    penalty = 0
    initial_budget = 1000
    fiat_account = initial_budget
    crypto_account = 0
    close_price = 0
    for index, row in data.iterrows():
        # Define the inputs including our current account status
        genome_input = list(row) + [fiat_account, crypto_account]
        prediction = genome.activate(genome_input)
        trading_decision = np.argmax(prediction[:3])
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
        # [0, x, 0]
        if trading_decision == 1:
            # We want to buy
            if fiat_account <= 0:
                # print('We dont have any fiat money left!')
                # Penalty for trying to trade without money
                fiat_account -= penalty
                continue
            # Make a trade for the open price
            if not fiat_trade_amount:
                # print('We cant make a trade for nothing!')
                fiat_account -= penalty
                continue

            # In case we want to sell more fiat than we have
            if fiat_account <= fiat_trade_amount:
                fiat_trade_amount = fiat_account

            amount_crypto_to_buy = fiat_trade_amount / open_price
            # Trading fees
            trading_fee_fiat = fiat_trade_amount * trading_fee_percent
            trading_fee_crypto = amount_crypto_to_buy * trading_fee_percent
            if fiat_account < trading_fee_fiat and crypto_account < trading_fee_crypto:
                # print('We cant pay the trading fees!')
                continue

            crypto_account += amount_crypto_to_buy
            fiat_account -= fiat_trade_amount
            if fiat_account > trading_fee_fiat:
                fiat_account -= trading_fee_fiat
            else:
                crypto_account -= trading_fee_crypto

        # [0, 0, x]
        elif trading_decision == 2:
            # We want to sell
            if crypto_account <= 0:
                # print('We dont have any crypto left!')
                fiat_account -= penalty
                continue
            # Make a trade for the open price
            if not fiat_trade_amount:
                # print('We cant make a trade for nothing!')
                fiat_account -= penalty
                continue

            amount_crypto_to_sell = fiat_trade_amount / open_price
            # In case we want to sell more crypto than we have
            if crypto_account <= amount_crypto_to_sell:
                amount_crypto_to_sell = crypto_account

            # Trading fees
            trading_fee_fiat = amount_crypto_to_sell * open_price * trading_fee_percent
            trading_fee_crypto = amount_crypto_to_sell * trading_fee_percent
            if fiat_account < trading_fee_fiat and crypto_account < trading_fee_crypto:
                # print('We cant pay the trading fees!')
                continue

            crypto_account -= amount_crypto_to_sell
            fiat_account += amount_crypto_to_sell * open_price
            if fiat_account > trading_fee_fiat:
                fiat_account -= trading_fee_fiat
            else:
                crypto_account -= trading_fee_crypto

        # [x, 0, 0]
        elif trading_decision == 0:
            # print('Hold your horses we hang tight till the next trade!')
            continue
    # Sell all open crypto for the last closing price
    if crypto_account > 0:
        fiat_account += crypto_account * close_price
        crypto_account = 0
    genome.fitness = fiat_account - initial_budget
    return genome


def on_generation(best, population):
    most_complex = None
    for genome in population.genomes.values():
        if not most_complex:
            most_complex = genome
        else:
            if sum(genome.complexity) > sum(most_complex.complexity):
                most_complex = genome
                print(f'Most complex genome is {genome.key} with {genome.complexity} and age {genome.generation}')

    trades = pandas.DataFrame(columns=['x', 'y', 'color', 'hover'])
    fiat_account = 1000
    crypto_account = 0
    close_price = 0
    for index, row in data.iterrows():
        # Define the inputs including our current account status
        genome_input = list(row) + [fiat_account, crypto_account]
        prediction = best.activate(genome_input)
        trading_decision = np.argmax(prediction[:3])
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
        # [0, x, 0]
        if trading_decision == 1:
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

            # Trading fees
            trading_fee_fiat = fiat_trade_amount * trading_fee_percent
            trading_fee_crypto = amount_crypto_to_buy * trading_fee_percent
            if fiat_account < trading_fee_fiat and crypto_account < trading_fee_crypto:
                # print('We cant pay the trading fees!')
                continue

            crypto_account += amount_crypto_to_buy
            fiat_account -= fiat_trade_amount
            if fiat_account > trading_fee_fiat:
                fiat_account -= trading_fee_fiat
            else:
                crypto_account -= trading_fee_crypto

            trades.loc[index] = {
                'x': index,
                'y': open_price,
                'color': 'green',
                'hover': f'Fiat: {fiat_trade_amount}<br>Crypto: {amount_crypto_to_buy}<br>Trading fee fiat: {trading_fee_fiat}',
            }
        # [0, 0, x]
        elif trading_decision == 2:
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

            # Trading fees
            trading_fee_fiat = amount_crypto_to_sell * open_price * trading_fee_percent
            trading_fee_crypto = amount_crypto_to_sell * trading_fee_percent
            if fiat_account < trading_fee_fiat and crypto_account < trading_fee_crypto:
                # print('We cant pay the trading fees!')
                continue

            crypto_account -= amount_crypto_to_sell
            fiat_account += amount_crypto_to_sell * open_price
            if fiat_account > trading_fee_fiat:
                fiat_account -= trading_fee_fiat
            else:
                crypto_account -= trading_fee_crypto

            trades.loc[index] = {
                'x': index,
                'y': open_price,
                'color': 'red',
                'hover': f'Fiat: {amount_crypto_to_sell * open_price}<br>Crypto: {amount_crypto_to_sell}<br>Trading fee fiat: {trading_fee_fiat}',
            }
        # [x, 0, 0]
        elif trading_decision == 0:
            # print('Hold your horses we hang tight till the next trade!')
            continue

    # Sell all open crypto for the last closing price
    if crypto_account > 0:
        fiat_trade_amount = crypto_account * close_price
        fiat_account += fiat_trade_amount
        # Trading fees
        trading_fee_fiat = fiat_trade_amount * trading_fee_percent
        trades.loc[data.index[-1]] = [
            data.index[-1],
            close_price,
            'red',
            f'Fiat: {fiat_trade_amount}<br>Crypto: {crypto_account}<br>Trading fees fiat: {trading_fee_fiat}',
        ]
    try:
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=data.index,
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    increasing_line_color='cyan',
                    decreasing_line_color='gray',
                ),
                go.Scatter(
                    x=trades.x,
                    y=trades.y,
                    mode='markers',
                    marker=dict(
                        color=trades.color,
                    ),
                    hovertext=trades.hover,
                    hoverlabel=dict(namelength=0),
                    hovertemplate='%{hovertext}'
                ),
            ])
    except:
        import pdb;
        pdb.set_trace()
    fig.update_yaxes(fixedrange=False)
    fig.write_html("./tradingpretrain.html")
    best_bot_outfile = open('15min-candle-trading-best-bot', 'wb')
    pickle.dump(best, best_bot_outfile)
    best_bot_outfile.close()


def on_success(best):
    best_bot_outfile = open('15min-candle-trading-best-bot', 'wb')
    population_outfile = open('15min-candle-trading-population', 'wb')
    pickle.dump(best, best_bot_outfile)
    pickle.dump(population, population_outfile)
    best_bot_outfile.close()
    population_outfile.close()


def run():
    multiprocessing.freeze_support()
    population.run(compute_fitness, on_success, on_generation, generations=10000)


if __name__ == '__main__':
    run()
    # pid = "/tmp/tradingpretrain.pid"
    # logger = logging.getLogger(__name__)
    # logger.setLevel(logging.DEBUG)
    # logger.propagate = False
    # fh = logging.FileHandler("/tmp/test.log", "w")
    # fh.setLevel(logging.DEBUG)
    # logger.addHandler(fh)
    # keep_fds = [fh.stream.fileno()]
    #
    # daemon = Daemonize(app="test_app", pid=pid, action=run, keep_fds=keep_fds)
    # daemon.start()
