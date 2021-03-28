import os
import pickle

import pandas
import plotly.graph_objects as go

dir_path = os.path.dirname(os.path.realpath(__file__))
data = pandas.read_pickle(os.path.join(dir_path, 'etheur-15min-candles-2.pkl'))
data = data.drop(['time', 'vwap', 'count'], axis=1)
data = data[(data.index >= "2021-03-12") & (data.index < "2021-03-13")]
data.sort_index(inplace=True)


def test(best):
    trades = pandas.DataFrame(columns=['x', 'y', 'color', 'hover'])
    for index, row in data.iterrows():
        # Define the inputs including our current account status
        genome_input = list(row)
        prediction = best.activate(genome_input)
        # Needs 0-1 activation function for outputs
        price_prediction = max(-1.0, min(1.0, prediction[0]))  # clamped
        trading_decision = 'buy' if price_prediction > 0 else 'sell' if price_prediction < 0 else 'hold'

        open_price = row.get('open')
        if trading_decision == 'buy':
            trades.loc[index] = {
                'x': index,
                'y': open_price,
                'color': 'green',
                'hover': f'Prediction: {price_prediction}',
            }
        elif trading_decision == 'sell':
            trades.loc[index] = {
                'x': index,
                'y': open_price,
                'color': 'red',
                'hover': f'Prediction: {price_prediction}',
            }
        elif trading_decision == 'hold':
            continue

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
    fig.write_html(os.path.join(dir_path, 'predict-price.html'))


if __name__ == '__main__':
    infile = open('15min-candle-predict-price-best-bot', 'rb')
    best = pickle.load(infile)
    infile.close()
    test(best)
