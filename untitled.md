import numpy as np
import pandas as pd
from pathlib import Path
%matplotlib inline

#Return Forecasting: Read Historical Daily Yen Futures Data
yen_futures = pd.read_csv(
    Path("yen.csv"), index_col="Date", infer_datetime_format=True, parse_dates=True
)
yen_futures.head()