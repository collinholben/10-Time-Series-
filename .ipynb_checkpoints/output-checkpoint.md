```python
import numpy as np
import pandas as pd
from pathlib import Path
%matplotlib inline
```

# Return Forecasting: Read Historical Daily Yen Futures Data
In this notebook, you will load historical Dollar-Yen exchange rate futures data and apply time series analysis and modeling to determine whether there is any predictable behavior.


```python
# Futures contract on the Yen-dollar exchange rate:
# This is the continuous chain of the futures contracts that are 1 month to expiration
yen_futures = pd.read_csv(
    Path("yen.csv"), index_col="Date", infer_datetime_format=True, parse_dates=True
)
yen_futures.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Last</th>
      <th>Change</th>
      <th>Settle</th>
      <th>Volume</th>
      <th>Previous Day Open Interest</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1976-08-02</th>
      <td>3398.0</td>
      <td>3401.0</td>
      <td>3398.0</td>
      <td>3401.0</td>
      <td>NaN</td>
      <td>3401.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1976-08-03</th>
      <td>3401.0</td>
      <td>3401.0</td>
      <td>3401.0</td>
      <td>3401.0</td>
      <td>NaN</td>
      <td>3401.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1976-08-04</th>
      <td>3401.0</td>
      <td>3401.0</td>
      <td>3401.0</td>
      <td>3401.0</td>
      <td>NaN</td>
      <td>3401.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1976-08-05</th>
      <td>3401.0</td>
      <td>3401.0</td>
      <td>3401.0</td>
      <td>3401.0</td>
      <td>NaN</td>
      <td>3401.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1976-08-06</th>
      <td>3401.0</td>
      <td>3401.0</td>
      <td>3401.0</td>
      <td>3401.0</td>
      <td>NaN</td>
      <td>3401.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Trim the dataset to begin on January 1st, 1990
yen_futures = yen_futures.loc["1990-01-01":, :]
yen_futures.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Last</th>
      <th>Change</th>
      <th>Settle</th>
      <th>Volume</th>
      <th>Previous Day Open Interest</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1990-01-02</th>
      <td>6954.0</td>
      <td>6954.0</td>
      <td>6835.0</td>
      <td>6847.0</td>
      <td>NaN</td>
      <td>6847.0</td>
      <td>48336.0</td>
      <td>51473.0</td>
    </tr>
    <tr>
      <th>1990-01-03</th>
      <td>6877.0</td>
      <td>6910.0</td>
      <td>6865.0</td>
      <td>6887.0</td>
      <td>NaN</td>
      <td>6887.0</td>
      <td>38206.0</td>
      <td>53860.0</td>
    </tr>
    <tr>
      <th>1990-01-04</th>
      <td>6937.0</td>
      <td>7030.0</td>
      <td>6924.0</td>
      <td>7008.0</td>
      <td>NaN</td>
      <td>7008.0</td>
      <td>49649.0</td>
      <td>55699.0</td>
    </tr>
    <tr>
      <th>1990-01-05</th>
      <td>6952.0</td>
      <td>6985.0</td>
      <td>6942.0</td>
      <td>6950.0</td>
      <td>NaN</td>
      <td>6950.0</td>
      <td>29944.0</td>
      <td>53111.0</td>
    </tr>
    <tr>
      <th>1990-01-08</th>
      <td>6936.0</td>
      <td>6972.0</td>
      <td>6936.0</td>
      <td>6959.0</td>
      <td>NaN</td>
      <td>6959.0</td>
      <td>19763.0</td>
      <td>52072.0</td>
    </tr>
  </tbody>
</table>
</div>



 # Return Forecasting: Initial Time-Series Plotting

 Start by plotting the "Settle" price. Do you see any patterns, long-term and/or short?


```python
# Plot just the "Settle" column from the dataframe:
yen_settle_prices = yen_futures['Settle'].plot(figsize=(12,8), title= "Yen Futures Settle Prices")
```


    
![png](output_files/output_6_0.png)
    


---

# Decomposition Using a Hodrick-Prescott Filter

 Using a Hodrick-Prescott Filter, decompose the Settle price into a trend and noise.


```python
import statsmodels.api as sm

# Apply the Hodrick-Prescott Filter by decomposing the "Settle" price into two separate series:
ts_noise, ts_trend = sm.tsa.filters.hpfilter(yen_futures['Settle'])


```


```python
# Create a dataframe of just the settle price, and add columns for "noise" and "trend" series from above:
df = pd.DataFrame(ts_noise)
df1 = pd.DataFrame(ts_trend)

settle_decompose = pd.concat([yen_futures['Settle'],df,df1], axis = 1, join = 'inner')
settle_decomposed = settle_decompose.rename(columns={"Settle_cycle" : "noise",
                                                      "Settle_trend" : "trend"})
settle_decomposed.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Settle</th>
      <th>noise</th>
      <th>trend</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1990-01-02</th>
      <td>6847.0</td>
      <td>-61.503967</td>
      <td>6908.503967</td>
    </tr>
    <tr>
      <th>1990-01-03</th>
      <td>6887.0</td>
      <td>-21.799756</td>
      <td>6908.799756</td>
    </tr>
    <tr>
      <th>1990-01-04</th>
      <td>7008.0</td>
      <td>98.942896</td>
      <td>6909.057104</td>
    </tr>
    <tr>
      <th>1990-01-05</th>
      <td>6950.0</td>
      <td>40.776052</td>
      <td>6909.223948</td>
    </tr>
    <tr>
      <th>1990-01-08</th>
      <td>6959.0</td>
      <td>49.689938</td>
      <td>6909.310062</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot the Settle Price vs. the Trend for 2015 to the present
settle_vs_trend = settle_decomposed.drop(columns=["noise"], axis='1', inplace=False)
settle_vs_trend.plot(figsize=(10,8), title = "Settle vs. Trend")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1e325f68288>




    
![png](output_files/output_12_1.png)
    



```python
# Plot the Settle Noise
ts_noise.plot(title = "Noise", figsize=(10,8))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1e32603a608>




    
![png](output_files/output_13_1.png)
    


---

# Forecasting Returns using an ARMA Model

Using futures Settle *Returns*, estimate an ARMA model

1. ARMA: Create an ARMA model and fit it to the returns data. Note: Set the AR and MA ("p" and "q") parameters to p=2 and q=1: order=(2, 1).
2. Output the ARMA summary table and take note of the p-values of the lags. Based on the p-values, is the model a good fit (p < 0.05)?
3. Plot the 5-day forecast of the forecasted returns (the results forecast from ARMA model)


```python
# Create a series using "Settle" price percentage returns, drop any nan"s, and check the results:
# (Make sure to multiply the pct_change() results by 100)
# In this case, you may have to replace inf, -inf values with np.nan"s
returns = (yen_futures[["Settle"]].pct_change() * 100)
returns = returns.replace(-np.inf, np.nan).dropna()
returns.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Settle</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-10-09</th>
      <td>-0.410601</td>
    </tr>
    <tr>
      <th>2019-10-10</th>
      <td>-0.369458</td>
    </tr>
    <tr>
      <th>2019-10-11</th>
      <td>-0.564304</td>
    </tr>
    <tr>
      <th>2019-10-14</th>
      <td>0.151335</td>
    </tr>
    <tr>
      <th>2019-10-15</th>
      <td>-0.469509</td>
    </tr>
  </tbody>
</table>
</div>




```python
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA
# Estimate and ARMA model using statsmodels (use order=(2, 1))

model = ARMA(returns.values, order=(2,1))

# Fit the model and assign it to a variable called results
results = model.fit()
```


```python
# Output model summary results:
results.summary()
```




<table class="simpletable">
<caption>ARMA Model Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>y</td>        <th>  No. Observations:  </th>   <td>7514</td>   
</tr>
<tr>
  <th>Model:</th>            <td>ARMA(2, 1)</td>    <th>  Log Likelihood     </th> <td>-7894.071</td>
</tr>
<tr>
  <th>Method:</th>             <td>css-mle</td>     <th>  S.D. of innovations</th>   <td>0.692</td>  
</tr>
<tr>
  <th>Date:</th>          <td>Sat, 05 Dec 2020</td> <th>  AIC                </th> <td>15798.142</td>
</tr>
<tr>
  <th>Time:</th>              <td>22:49:27</td>     <th>  BIC                </th> <td>15832.765</td>
</tr>
<tr>
  <th>Sample:</th>                <td>0</td>        <th>  HQIC               </th> <td>15810.030</td>
</tr>
<tr>
  <th></th>                       <td> </td>        <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>        <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>   <td>    0.0064</td> <td>    0.008</td> <td>    0.804</td> <td> 0.421</td> <td>   -0.009</td> <td>    0.022</td>
</tr>
<tr>
  <th>ar.L1.y</th> <td>   -0.3063</td> <td>    1.278</td> <td>   -0.240</td> <td> 0.811</td> <td>   -2.811</td> <td>    2.198</td>
</tr>
<tr>
  <th>ar.L2.y</th> <td>   -0.0019</td> <td>    0.019</td> <td>   -0.099</td> <td> 0.921</td> <td>   -0.040</td> <td>    0.036</td>
</tr>
<tr>
  <th>ma.L1.y</th> <td>    0.2948</td> <td>    1.278</td> <td>    0.231</td> <td> 0.818</td> <td>   -2.210</td> <td>    2.799</td>
</tr>
</table>
<table class="simpletable">
<caption>Roots</caption>
<tr>
    <td></td>   <th>            Real</th>  <th>         Imaginary</th> <th>         Modulus</th>  <th>        Frequency</th>
</tr>
<tr>
  <th>AR.1</th> <td>          -3.3338</td> <td>          +0.0000j</td> <td>           3.3338</td> <td>           0.5000</td>
</tr>
<tr>
  <th>AR.2</th> <td>        -157.0042</td> <td>          +0.0000j</td> <td>         157.0042</td> <td>           0.5000</td>
</tr>
<tr>
  <th>MA.1</th> <td>          -3.3926</td> <td>          +0.0000j</td> <td>           3.3926</td> <td>           0.5000</td>
</tr>
</table>




```python
# Plot the 5 Day Returns Forecast
results_df = pd.DataFrame(results.forecast(steps=5)[0]).plot(title="5 Day Return Forecast")
```


    
![png](output_files/output_20_0.png)
    


---

# Forecasting the Settle Price using an ARIMA Model

 1. Using the *raw* Yen **Settle Price**, estimate an ARIMA model.
     1. Set P=5, D=1, and Q=1 in the model (e.g., ARIMA(df, order=(5,1,1))
     2. P= # of Auto-Regressive Lags, D= # of Differences (this is usually =1), Q= # of Moving Average Lags
 2. Output the ARIMA summary table and take note of the p-values of the lags. Based on the p-values, is the model a good fit (p < 0.05)?
 3. Construct a 5 day forecast for the Settle Price. What does the model forecast will happen to the Japanese Yen in the near term?


```python
from statsmodels.tsa.arima_model import ARIMA
p=5
d=1
q=1
# Estimate and ARIMA Model:
# Hint: ARIMA(df, order=(p, d, q))
arima = ARIMA(yen_futures[['Settle']], order=(p, d, q))

# Fit the model
results = arima.fit()

```

    C:\Users\colli\.conda\envs\alpacaenv\lib\site-packages\statsmodels\tsa\base\tsa_model.py:218: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      ' ignored when e.g. forecasting.', ValueWarning)
    C:\Users\colli\.conda\envs\alpacaenv\lib\site-packages\statsmodels\tsa\base\tsa_model.py:218: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      ' ignored when e.g. forecasting.', ValueWarning)
    


```python
# Output model summary results:
results.summary()
```




<table class="simpletable">
<caption>ARIMA Model Results</caption>
<tr>
  <th>Dep. Variable:</th>     <td>D.Settle</td>     <th>  No. Observations:  </th>    <td>7514</td>   
</tr>
<tr>
  <th>Model:</th>          <td>ARIMA(5, 1, 1)</td>  <th>  Log Likelihood     </th> <td>-41944.619</td>
</tr>
<tr>
  <th>Method:</th>             <td>css-mle</td>     <th>  S.D. of innovations</th>   <td>64.281</td>  
</tr>
<tr>
  <th>Date:</th>          <td>Sat, 05 Dec 2020</td> <th>  AIC                </th>  <td>83905.238</td>
</tr>
<tr>
  <th>Time:</th>              <td>22:49:28</td>     <th>  BIC                </th>  <td>83960.635</td>
</tr>
<tr>
  <th>Sample:</th>                <td>1</td>        <th>  HQIC               </th>  <td>83924.259</td>
</tr>
<tr>
  <th></th>                       <td> </td>        <th>                     </th>      <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
         <td></td>           <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>          <td>    0.3159</td> <td>    0.700</td> <td>    0.451</td> <td> 0.652</td> <td>   -1.056</td> <td>    1.688</td>
</tr>
<tr>
  <th>ar.L1.D.Settle</th> <td>    0.2820</td> <td>    0.699</td> <td>    0.403</td> <td> 0.687</td> <td>   -1.089</td> <td>    1.653</td>
</tr>
<tr>
  <th>ar.L2.D.Settle</th> <td>    0.0007</td> <td>    0.016</td> <td>    0.043</td> <td> 0.966</td> <td>   -0.030</td> <td>    0.032</td>
</tr>
<tr>
  <th>ar.L3.D.Settle</th> <td>   -0.0126</td> <td>    0.012</td> <td>   -1.032</td> <td> 0.302</td> <td>   -0.037</td> <td>    0.011</td>
</tr>
<tr>
  <th>ar.L4.D.Settle</th> <td>   -0.0137</td> <td>    0.015</td> <td>   -0.889</td> <td> 0.374</td> <td>   -0.044</td> <td>    0.016</td>
</tr>
<tr>
  <th>ar.L5.D.Settle</th> <td>   -0.0012</td> <td>    0.018</td> <td>   -0.064</td> <td> 0.949</td> <td>   -0.036</td> <td>    0.034</td>
</tr>
<tr>
  <th>ma.L1.D.Settle</th> <td>   -0.2970</td> <td>    0.699</td> <td>   -0.425</td> <td> 0.671</td> <td>   -1.668</td> <td>    1.074</td>
</tr>
</table>
<table class="simpletable">
<caption>Roots</caption>
<tr>
    <td></td>   <th>            Real</th>  <th>         Imaginary</th> <th>         Modulus</th>  <th>        Frequency</th>
</tr>
<tr>
  <th>AR.1</th> <td>           1.8915</td> <td>          -1.3788j</td> <td>           2.3407</td> <td>          -0.1003</td>
</tr>
<tr>
  <th>AR.2</th> <td>           1.8915</td> <td>          +1.3788j</td> <td>           2.3407</td> <td>           0.1003</td>
</tr>
<tr>
  <th>AR.3</th> <td>          -2.2689</td> <td>          -3.0213j</td> <td>           3.7784</td> <td>          -0.3525</td>
</tr>
<tr>
  <th>AR.4</th> <td>          -2.2689</td> <td>          +3.0213j</td> <td>           3.7784</td> <td>           0.3525</td>
</tr>
<tr>
  <th>AR.5</th> <td>         -11.0308</td> <td>          -0.0000j</td> <td>          11.0308</td> <td>          -0.5000</td>
</tr>
<tr>
  <th>MA.1</th> <td>           3.3671</td> <td>          +0.0000j</td> <td>           3.3671</td> <td>           0.0000</td>
</tr>
</table>




```python
# Plot the 5 Day Price Forecast
pd.DataFrame(results.forecast(steps=5)[0]).plot(title="5 Day Futures Price forecast")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1e326711f48>




    
![png](output_files/output_26_1.png)
    




---

# Volatility Forecasting with GARCH

Rather than predicting returns, let's forecast near-term **volatility** of Japanese Yen futures returns. Being able to accurately predict volatility will be extremely useful if we want to trade in derivatives or quantify our maximum loss.
 
Using futures Settle *Returns*, estimate an GARCH model

1. GARCH: Create an GARCH model and fit it to the returns data. Note: Set the parameters to p=2 and q=1: order=(2, 1).
2. Output the GARCH summary table and take note of the p-values of the lags. Based on the p-values, is the model a good fit (p < 0.05)?
3. Plot the 5-day forecast of the volatility.


```python
!pip install arch 
import arch 
from arch import arch_model
```

    Requirement already satisfied: arch in c:\users\colli\.conda\envs\alpacaenv\lib\site-packages (4.15)
    Requirement already satisfied: cython>=0.29.14 in c:\users\colli\.conda\envs\alpacaenv\lib\site-packages (from arch) (0.29.21)
    Requirement already satisfied: scipy>=1.0.1 in c:\users\colli\.conda\envs\alpacaenv\lib\site-packages (from arch) (1.5.0)
    Requirement already satisfied: property-cached>=1.6.3 in c:\users\colli\.conda\envs\alpacaenv\lib\site-packages (from arch) (1.6.4)
    Requirement already satisfied: statsmodels>=0.9 in c:\users\colli\.conda\envs\alpacaenv\lib\site-packages (from arch) (0.11.1)
    Requirement already satisfied: pandas>=0.23 in c:\users\colli\.conda\envs\alpacaenv\lib\site-packages (from arch) (1.0.5)
    Requirement already satisfied: numpy>=1.14 in c:\users\colli\.conda\envs\alpacaenv\lib\site-packages (from arch) (1.17.0)
    Requirement already satisfied: patsy>=0.5 in c:\users\colli\.conda\envs\alpacaenv\lib\site-packages (from statsmodels>=0.9->arch) (0.5.1)
    Requirement already satisfied: pytz>=2017.2 in c:\users\colli\.conda\envs\alpacaenv\lib\site-packages (from pandas>=0.23->arch) (2020.1)
    Requirement already satisfied: python-dateutil>=2.6.1 in c:\users\colli\.conda\envs\alpacaenv\lib\site-packages (from pandas>=0.23->arch) (2.8.1)
    Requirement already satisfied: six in c:\users\colli\.conda\envs\alpacaenv\lib\site-packages (from patsy>=0.5->statsmodels>=0.9->arch) (1.15.0)
    


```python
# Estimate a GARCH model:
garch = arch_model(returns, mean="Zero", vol="GARCH", p=2, q=1)

# Fit the model
model = garch.fit(disp="off")

```


```python
# Summarize the model results
model.summary()
```




<table class="simpletable">
<caption>Zero Mean - GARCH Model Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>Settle</td>       <th>  R-squared:         </th>  <td>   0.000</td> 
</tr>
<tr>
  <th>Mean Model:</th>         <td>Zero Mean</td>     <th>  Adj. R-squared:    </th>  <td>   0.000</td> 
</tr>
<tr>
  <th>Vol Model:</th>            <td>GARCH</td>       <th>  Log-Likelihood:    </th> <td>  -7461.93</td>
</tr>
<tr>
  <th>Distribution:</th>        <td>Normal</td>       <th>  AIC:               </th> <td>   14931.9</td>
</tr>
<tr>
  <th>Method:</th>        <td>Maximum Likelihood</td> <th>  BIC:               </th> <td>   14959.6</td>
</tr>
<tr>
  <th></th>                        <td></td>          <th>  No. Observations:  </th>    <td>7514</td>   
</tr>
<tr>
  <th>Date:</th>           <td>Sat, Dec 05 2020</td>  <th>  Df Residuals:      </th>    <td>7510</td>   
</tr>
<tr>
  <th>Time:</th>               <td>22:49:31</td>      <th>  Df Model:          </th>      <td>4</td>    
</tr>
</table>
<table class="simpletable">
<caption>Volatility Model</caption>
<tr>
      <td></td>        <th>coef</th>     <th>std err</th>      <th>t</th>       <th>P>|t|</th>      <th>95.0% Conf. Int.</th>   
</tr>
<tr>
  <th>omega</th>    <td>4.2896e-03</td> <td>2.057e-03</td> <td>    2.085</td> <td>3.708e-02</td>  <td>[2.571e-04,8.322e-03]</td>
</tr>
<tr>
  <th>alpha[1]</th> <td>    0.0381</td> <td>1.282e-02</td> <td>    2.970</td> <td>2.974e-03</td>  <td>[1.295e-02,6.321e-02]</td>
</tr>
<tr>
  <th>alpha[2]</th>   <td>0.0000</td>   <td>1.703e-02</td>   <td>0.000</td>   <td>    1.000</td> <td>[-3.338e-02,3.338e-02]</td>
</tr>
<tr>
  <th>beta[1]</th>  <td>    0.9536</td> <td>1.420e-02</td> <td>   67.135</td>   <td>0.000</td>      <td>[  0.926,  0.981]</td>  
</tr>
</table><br/><br/>Covariance estimator: robust




```python
# Find the last day of the dataset
last_day = returns.index.max().strftime('%Y-%m-%d')
last_day
```




    '2019-10-15'




```python
# Create a 5 day forecast of volatility
forecast_horizon = 5
# Start the forecast using the last_day calculated above
forecasts = model.forecast(start = last_day, horizon = forecast_horizon)
forecasts
```




    <arch.univariate.base.ARCHModelForecast at 0x1e32302f188>




```python
# Annualize the forecast
intermediate = np.sqrt(forecasts.variance.dropna() * 252)
intermediate.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>h.1</th>
      <th>h.2</th>
      <th>h.3</th>
      <th>h.4</th>
      <th>h.5</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-10-15</th>
      <td>7.434048</td>
      <td>7.475745</td>
      <td>7.516867</td>
      <td>7.557426</td>
      <td>7.597434</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Transpose the forecast so that it is easier to plot
final = intermediate.dropna().T
final.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Date</th>
      <th>2019-10-15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>h.1</th>
      <td>7.434048</td>
    </tr>
    <tr>
      <th>h.2</th>
      <td>7.475745</td>
    </tr>
    <tr>
      <th>h.3</th>
      <td>7.516867</td>
    </tr>
    <tr>
      <th>h.4</th>
      <td>7.557426</td>
    </tr>
    <tr>
      <th>h.5</th>
      <td>7.597434</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot the final forecast
final.plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1e325f2bb08>




    
![png](output_files/output_37_1.png)
    


---

# Conclusions

Based on your time series analysis, would you buy the yen now?

Is the risk of the yen expected to increase or decrease?

Based on the model evaluation, would you feel confident in using these models for trading?

just based on this anylsis, the yen is more bullish than bearish.

the risk of yen is expected to increase

no i would not feel confident in using these models, maybe only as supplimental metrics to reinforce or contradic a security I was looking to invest in.
