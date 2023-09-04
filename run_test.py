import pandas as pd
from darts.models.forecasting.nhits import NHiTSModel
import torch

from helpers import predict, load_agent, quality, clip
from preprocess import preprocess_stats


def optimize_params(data: pd.DataFrame) -> pd.DataFrame:
    """
    Run and evaluate agent.

    :param data:        raw observations in pandas DataFrame
    :return:            result saves to the same path as input
    """
    columns = ['Cell ID', 'LAC', 'HR Usage Rate', 'TCH Blocking Rate, BH', 'Number of Available\nTCH',
               'TCH Traffic (Erl), BH', 'Lower_limit', 'Upper_limit']

    df = preprocess_stats(data, columns)
    obs_array = df.drop(columns=['Cell ID', 'LAC']).values

    agent = load_agent('sac_last_60_50d_exp-r.pt', 'pt')
    state_predictor = NHiTS.load_from_checkpoint("state_predictor/nhits_1.38.pth.tar")

    # # 'HR Usage Rate', 'TCH Blocking Rate, BH'
    # self.current_state = series[randint(0, len(series))].head(n_past)
    # # 'Number of Available\nTCH', 'TCH Traffic (Erl), BH', 'Param 1',  'Param 2'
    # self.cov = covariates[0].head(n_past)

    # n for number of states to predict
    state_predictor.predict(n=1, series=current_state, past_covariates=cov)

    lower_limits = []
    upper_limits = []
    qualities = []

    for row in obs_array:
        a1, a2 = predict(row, agent)
        lower = clip(int(row[-2] + a1 * 30))
        upper = clip(int(row[-1] + a2 * 30))

        # Compute quality
        qualities.append(
            quality(blocking=row[1], ch=row[2], traffic=row[3], param1=row[-2], param2=row[-1], prparam1=lower,
                    prparam2=upper)
        )
        # df["Quality Rate"] = 1 - (2*df['HR Usage Rate']/100 + np.log(df['TCH Blocking Rate, BH'] + 1))/(1 + np.log(101))

        lower_limits.append(lower)
        upper_limits.append(upper)

    df['Lower_limit_Gen'], df['Upper_limit_Gen'], df['Limit_quality_Gen'] = lower_limits, upper_limits, qualities

    return df


if __name__ == '__main__':

    # how to use
    df = optimize_params(pd.read_excel('data/GBTS_TOTAL_20220522.xlsm'))

    df.to_excel('quality_results.xlsm')
