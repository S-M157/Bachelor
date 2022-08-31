import pandas as pd

from helpers import predict, load_agent, bad_decision, clip
from preprocess import preprocess_stats


def optimize_params(data: pd.DataFrame) -> pd.DataFrame:
    """
    function for optimization lower and upper FR usage limit

    :param data:        stats in pandas DataFrame 
    :return:            result saves to the same path as input
    """
    columns = ['Cell ID', 'LAC', 'HR Usage Rate', 'TCH Blocking Rate, BH', 'Number of Available\nTCH',
               'TCH Traffic (Erl), BH', 'Lower_limit', 'Upper_limit']

    df = preprocess_stats(path_to_stats, columns)
    obs_array = df.drop(columns=['Cell ID', 'LAC']).values

    agent = load_agent('sac_last_60_50d_exp-r.pt', 'pt')
    lower_limits = []
    upper_limits = []

    for row in obs_array:
        a1, a2 = predict(row, agent)
        lower = clip(int(row[-2] + a1 * 30))
        upper = clip(int(row[-1] + a2 * 30))
        # don't change if decision is bad
        if bad_decision(blocking=row[1], channels=row[2], traffic=row[3], param1=row[-2], param2=row[-1],
                        prparam1=lower, prparam2=upper):
            lower, upper = row[-2], row[-1]

        lower_limits.append(lower)
        upper_limits.append(upper)
        
    df['Lower_limit_Gen'], df['Upper_limit_Gen'] = lower_limits, upper_limits

    return df

# how to use
# optimize_params('data/GBTS_TOTAL_20220522.xlsm')
