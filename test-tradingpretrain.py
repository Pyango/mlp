import pickle

import numpy as np
import pandas
import plotly.graph_objects as go

data = pandas.read_pickle("./etheur-15min-candles-1.pkl")
data.sort_index(inplace=True)
trade_size_fiat = 100
trading_fee_percent = 0.0016


def test(best):
    trades = pandas.DataFrame(columns=['x', 'y', 'color', 'hover'])
    fiat_account = 1000
    crypto_account = 0
    total_trading_fees = 0
    close_price = 0
    for index, row in data.iterrows():
        # Define the inputs including our current account status
        genome_input = list(row) + [fiat_account, crypto_account]
        prediction = best.activate(genome_input)
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
        # [0, x, 0]
        if trading_decision == 1:
            # We want to buy
            if fiat_account <= 0:
                print('We dont have any fiat money left!')
                continue
            # Make a trade for the open price
            if not fiat_trade_amount:
                print('We cant make a trade for nothing!')
                continue

            # In case we want to sell more fiat than we have
            if fiat_account <= fiat_trade_amount:
                fiat_trade_amount = fiat_account

            amount_crypto_to_buy = fiat_trade_amount / open_price

            # Trading fees
            trading_fee_fiat = fiat_trade_amount * trading_fee_percent
            trading_fee_crypto = amount_crypto_to_buy * trading_fee_percent
            if fiat_account < trading_fee_fiat and crypto_account < trading_fee_crypto:
                print('We cant pay the trading fees!')
                continue

            crypto_account += amount_crypto_to_buy
            fiat_account -= fiat_trade_amount
            if fiat_account > trading_fee_fiat:
                fiat_account -= trading_fee_fiat
            else:
                crypto_account -= trading_fee_crypto

            print(f'Buy {amount_crypto_to_buy} in crypto for {fiat_trade_amount} fiat')
            total_trading_fees += trading_fee_fiat
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
                print('We dont have any crypto left!')
                continue
            # Make a trade for the open price
            if not fiat_trade_amount:
                print('We cant make a trade for nothing!')
                continue

            amount_crypto_to_sell = fiat_trade_amount / open_price
            # In case we want to sell more crypto than we have
            if crypto_account <= amount_crypto_to_sell:
                amount_crypto_to_sell = crypto_account

            # Trading fees
            trading_fee_fiat = amount_crypto_to_sell * open_price * trading_fee_percent
            trading_fee_crypto = amount_crypto_to_sell * trading_fee_percent
            if fiat_account < trading_fee_fiat and crypto_account < trading_fee_crypto:
                print('We cant pay the trading fees!')
                continue

            crypto_account -= amount_crypto_to_sell
            fiat_account += amount_crypto_to_sell * open_price
            if fiat_account > trading_fee_fiat:
                fiat_account -= trading_fee_fiat
            else:
                crypto_account -= trading_fee_crypto

            total_trading_fees += trading_fee_fiat

            print(f'Sell {amount_crypto_to_sell} in crypto for {amount_crypto_to_sell * open_price} fiat')
            trades.loc[index] = {
                'x': index,
                'y': open_price,
                'color': 'red',
                'hover': f'Fiat: {amount_crypto_to_sell * open_price}<br>Crypto: {amount_crypto_to_sell}<br>Trading fee fiat: {trading_fee_fiat}',
            }
        # [x, 0, 0]
        elif trading_decision == 0:
            print('Hold your horses we hang tight till the next trade!')
            continue

    # Sell all open crypto for the last closing price
    if crypto_account > 0:
        fiat_trade_amount = crypto_account * close_price
        fiat_account += fiat_trade_amount
        # Trading fees
        trading_fee_fiat = fiat_trade_amount * trading_fee_percent
        total_trading_fees += trading_fee_fiat
        print(f'Buy {crypto_account} in crypto for {fiat_trade_amount} fiat')
        trades.loc[data.index[-1]] = [
            data.index[-1],
            close_price,
            'red',
            f'Fiat: {fiat_trade_amount}<br>Crypto: {crypto_account}<br>Trading fees fiat: {trading_fee_fiat}',
        ]
    print(f'Your account contains {fiat_account} fiat')
    print(f'We made {len(trades)} trades')
    print(f'and paid {total_trading_fees} fiat in trading fees')
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
    fig.update_yaxes(fixedrange=False)
    fig.write_html("./test-tradingpretrain.html")


if __name__ == '__main__':
    infile = open('15min-candle-trading-best-bot', 'rb')
    best = pickle.load(infile)
    infile.close()
    test(best)
