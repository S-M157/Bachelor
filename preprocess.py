import pandas as pd
import numpy as np


def get_param(s):
    arr = np.array(
        [np.array(i.split('-'), dtype=int)[0:2] if len(np.unique(i.split('-'))) == 2 else [np.nan, np.nan] for i in s])
    return arr[:, 0], arr[:, 1]


def num(data):
    for i in data.columns:
        if "time" not in i.lower() and "name" not in i.lower() and "DATA" not in i:
            data[i] = pd.to_numeric(data[i])
        elif "time" in i.lower() or 'DATA' in i:
            data[i] = pd.to_datetime(data[i])
    return data


def preprocess_stats(data, cols):
    data.columns = data.iloc[1]
    data = data.drop(index=[0, 1])

    data = data.dropna(axis=0, how='all')
    data = data.dropna(axis=1, how='all')

    data['Lower_limit'], data['Upper_limit'] = get_param(data['Database HRACT1(порог HR)'])
    data = data.drop(columns=['Database HRACT1(порог HR)'])
    data = data[data['availability ratio'] == 100]
    data = data[cols]
    
    data = data.dropna()
    data = num(data)
    
    return data
 
