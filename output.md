```python
import numpy as np
import pandas as pd
from pathlib import Path
%matplotlib inline

###Return Forecasting: Initial Time-Series Plotting###
yen_settle_prices = yen_futures['Settle'].plot(figsize=(12,8), title= "Yen Futures Settle Prices")
 

###Decomposition Using a Hodrick-Prescott Filter###
import statsmodels.api as sm

# Apply the Hodrick-Prescott Filter by decomposing the "Settle" price into two separate series:
ts_noise, ts_trend = sm.tsa.filters.hpfilter(yen_futures['Settle'])

# Create a dataframe of just the settle price, and add columns for "noise" and "trend" series from above:
df = pd.DataFrame(ts_noise)
df1 = pd.DataFrame(ts_trend)

settle_decompose = pd.concat([yen_futures['Settle'],df,df1], axis = 1, join = 'inner')
settle_decomposed = settle_decompose.rename(columns={"Settle_cycle" : "noise",
                                                      "Settle_trend" : "trend"})
settle_decomposed.head()

# Plot the Settle Price vs. the Trend for 2015 to the present
settle_vs_trend = settle_decomposed.drop(columns=["noise"], axis='1', inplace=False)
settle_vs_trend.plot(figsize=(10,8), title = "Settle vs. Trend")

# Plot the Settle Noise
ts_noise.plot(title = "Noise", figsize=(10,8))


###Forecasting Returns using an ARMA Model###
# Create a series using "Settle" price percentage returns
returns = (yen_futures[["Settle"]].pct_change() * 100)
returns = returns.replace(-np.inf, np.nan).dropna()
returns.tail()