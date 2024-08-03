from twelvedata import TDClient
import pandas as pd
from datetime import datetime
import numpy as np
from scipy.stats import norm
"""
td = TDClient(apikey="eba7da3b24104ca594f061cb762cb8da")

df = td.time_series(symbol="AAPL", interval="1day", start_date="2018-01-01", end_date="2024-07-01").as_pandas()
"""

price = 107
strike = 130
free_rate = 0.0375
implied_vol = 0.5
maturity_date = "2024-08-05"

def days_between():
    date1 = datetime.strptime(datetime.today().strftime('%Y-%m-%d'), "%Y-%m-%d")
    maturity_date = datetime.strptime("2024-08-05", "%Y-%m-%d")
    return abs((maturity_date - date1).days)

def prob_of_z(z):
    return norm.cdf(z)

def d1():
    return (np.log(price / strike) + (free_rate + implied_vol**2 / 2) * (days_between()))

def d2():
    return d1() - implied_vol * days_between()

def call():
    return price*prob_of_z(d1()) - strike * np.exp(-free_rate*days_between()) * prob_of_z(d2())

def put():
    return -price*prob_of_z(d1()) + strike * np.exp(-free_rate*days_between()) * prob_of_z(d2())

print(call())