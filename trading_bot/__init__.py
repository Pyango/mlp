from datetime import timedelta, datetime
import krakenex
from pykrakenapi import KrakenAPI

api = krakenex.API()
k = KrakenAPI(api)
now = datetime.now()
last_interval = now - timedelta(days=30)
timestamp = datetime.timestamp(last_interval)

# Get the ETHEUR prices for the last few days
ohlc = k.get_ohlc_data("ETHEUR", since=timestamp, interval=15)
ohlc[0].to_pickle("../etheur-15min-candles-2.pkl")
