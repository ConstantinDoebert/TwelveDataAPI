from twelvedata import TDClient
import pandas as pd

td = TDClient(apikey="eba7da3b24104ca594f061cb762cb8da")

df = td.time_series(symbol="AAPL", interval="1day", start_date="2018-01-01", end_date="2024-07-01").as_pandas()
