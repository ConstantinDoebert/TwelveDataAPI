from datetime import datetime, timedelta
from scipy.stats import norm, ttest_1samp, t
import numpy as np
import requests as rq
import pandas as pd


key = "eba7da3b24104ca594f061cb762cb8da"


price_cache = {}

def get_price(ticker: str):
    if 'price' not in price_cache:
        price = rq.get(f"https://api.twelvedata.com/price?symbol={ticker}&apikey={key}")
        price_cache['price'] = float(price.json()["price"])
    return price_cache


def get_earnings_dates(ticker: str, num_intervals=3):
    earnings_dates = rq.get(f"https://api.twelvedata.com/earnings?symbol={ticker}&apikey={key}&outputsize=20").json()
    dates = []
    today = datetime.today().date()
    for item in earnings_dates["earnings"]:
        earnings_date = datetime.strptime(item['date'], "%Y-%m-%d").date()

        if earnings_date <= today:
            dates.append(datetime.strptime(item['date'], "%Y-%m-%d").date())

    dates = dates[:num_intervals]

    return dates


def get_time_series(ticker: str):
    dates = get_earnings_dates(ticker)
    start_dates = []
    start_dates = [date - timedelta(days=41) for date in dates]

    all_closing_prices = []
    
    for i in range(len(dates)):
        start_date = start_dates[i]
        end_date = dates[i]
        
        response = rq.get(
            f"https://api.twelvedata.com/time_series",
            params={
                "symbol": ticker,
                "interval": "1day",
                "apikey": key,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
            }
        )
        
        time_series = response.json()


        if "values" in time_series:
            closing_prices = [float(value["close"]) for value in time_series["values"]]
            all_closing_prices.append(closing_prices)
    

    max_length = max(len(prices) for prices in all_closing_prices)

    padded_closing_prices = [prices + [float('nan')] * (max_length - len(prices)) for prices in all_closing_prices]

    df = pd.DataFrame(padded_closing_prices).T
    df.columns = [f'Interval {i+1}' for i in range(len(dates))]

    df = df.iloc[::-1].reset_index(drop=True)
    
    return df


def analyze_returns(ticker: str):
    df = get_time_series(ticker)
    means, standard_devs, avg_return, sample_size = ([], [], [], [])

    for column in df:
        means.append(df[column].dropna().mean())
        standard_devs.append(np.sqrt(df[column].dropna().var())) # Bessel's correction 
        avg_return.append((df[column].dropna().iloc[-1] - df[column].dropna().iloc[0]) / df[column].dropna().iloc[0])
        sample_size.append(df[column].dropna().count())
    
    return means, standard_devs, avg_return, sample_size

def one_sample_t_test(ticker: str, alpha=0.1):
    """
    Tests if total return over period is significantly deviation from 0. One sided test.
    Level of signficance is set to 0.05 for standard.
    """
    values = analyze_returns(ticker)
    standard_devs = values[1]
    avg_return = values[2]
    sample_size = values[3]
    is_significant = []

    for sample in range(len(avg_return)):
        print(avg_return[sample], standard_devs[sample], sample_size[sample])
        if avg_return[sample] / standard_devs[sample] * np.sqrt(sample_size[sample]) < -t.cdf(1-alpha, df=sample_size[sample]):
            is_significant.append(True)
        else:
            is_significant.append(False)
    # for column in df:
    #     if ttest_1samp(df[column], popmean=0, alternative="greater", nan_policy="omit")[1] < alpha:
    #         is_significant.append(True)
    #     else:
    #         is_significant.append(False)
        
    return is_significant

print(one_sample_t_test("NVDA"))


strike = 130
free_rate = 0.0375
implied_vol = 0.5
maturity_date = "2024-08-05"


def days_between(maturity_date):
    date1 = datetime.today().date()
    maturity_date = datetime.strptime(maturity_date, "%Y-%m-%d")
    return abs((maturity_date - date1).days)

def prob_of_z(z):
    return norm.cdf(z)

def d1():
    return (np.log(price_cache["price"] / strike) + (free_rate + implied_vol**2 / 2) * (days_between(maturity_date)))

def d2():
    return d1() - implied_vol * days_between(maturity_date)

def call():
    return [price_cache["price"]*prob_of_z(d1()) - strike * np.exp(-free_rate*days_between(maturity_date)) * prob_of_z(d2())]

def put():
    return [-price_cache["price"]*prob_of_z(d1()) + strike * np.exp(-free_rate*days_between(maturity_date)) * prob_of_z(d2())]


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