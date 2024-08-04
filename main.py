from twelvedata import TDClient
from datetime import datetime
from scipy.stats import norm
import numpy as np
import requests as rq
import pandas as pd

key = "eba7da3b24104ca594f061cb762cb8da"

def get_price(ticker: str):
    price = rq.get(f"https://api.twelvedata.com/price?symbol={ticker}&apikey={key}")
    price = float(price.json()["price"])
    return price

# price = get_price("NVDA")

td = TDClient(apikey=key)

def get_time_series(ticker: str, num_intervals = 10):
    earnings_dates = rq.get(f"https://api.twelvedata.com/earnings?symbol={ticker}&apikey={key}&outputsize={num_intervals}").json()["earnings"]
    """for interval in range(num_intervals):
        df = td.time_series(symbol=ticker, 
                            interval="1day", 
                            start_date="2018-01-01", 
                            end_date="2024-07-01"
                            ).as_pandas()"""
    return earnings_dates

print(get_time_series("AAPL"))

strike = 130
free_rate = 0.0375
implied_vol = 0.5
maturity_date = "2024-08-05"

def days_between(maturity_date):
    date1 = datetime.strptime(datetime.today().strftime('%Y-%m-%d'), "%Y-%m-%d")
    maturity_date = datetime.strptime(maturity_date, "%Y-%m-%d")
    return abs((maturity_date - date1).days)

def prob_of_z(z):
    return norm.cdf(z)

def d1():
    return (np.log(price / strike) + (free_rate + implied_vol**2 / 2) * (days_between(maturity_date)))

def d2():
    return d1() - implied_vol * days_between(maturity_date)

def call():
    return [price*prob_of_z(d1()) - strike * np.exp(-free_rate*days_between(maturity_date)) * prob_of_z(d2())]

def put():
    return [-price*prob_of_z(d1()) + strike * np.exp(-free_rate*days_between(maturity_date)) * prob_of_z(d2())]


def simulate_stock_prices(initial_prices, expected_returns, volatilities, correlations, time_horizon):
    "np.random.seed(42)  # Set seed for reproducibility"
    num_assets = len(initial_prices) # number of different assets by number of input of initial prices (starting prices at t0)
    num_steps = int(time_horizon) # number of steps to run Monte Carlo-Simulation

    # Convert volatilities to Numpy array
    volatilities = np.array(volatilities) # 1 dimensional array; check with np object.ndim <- for example: volatilites.ndim = 1
    

    stock_prices = np.zeros((num_steps + 1, num_assets)) # create 2 dimensional array filled with zeros; nrow = trading days + 1, ncol = 1 for each asset
    stock_prices[0, :] = initial_prices # fill the first row with the initial prices at t0

    # Generate uncorrelated random numbers from a normal distribution
    z = np.random.normal(0, 1, size=(num_steps, num_assets))
    print("\n\n\n\n",z, "\n\n\n\n")

    for j in range(1, num_steps + 1): # for each row (trading day)
        for k in range(num_assets): # for each asset
            stock_prices[j, k] = stock_prices[j - 1, k] * (1+(expected_returns[k] + volatilities[k] * (correlations[k] * z[j-1,0] + z[j-1,k]*(1-correlations[k]**2)**0.5)))
            """
            1. calculate the stock price for t by looking at the value for t-1
            2. multiply the prior day's stock price with 
            """
    return stock_prices

# initial_prices = call()