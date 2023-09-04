import argparse
import warnings

import numpy as np
import pandas as pd
from darts.models.forecasting.nhits import NHiTSModel
from darts import TimeSeries
import torch
from typing import Callable
from tqdm import tqdm
from typing import List

from helpers import predict, load_agent, quality, clip
from preprocess import preprocess_stats
from rl.sim_enviroment import SimulatedCustomEnv

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.options import DataDriftOptions


def optimize_params(data: pd.DataFrame, preprocess: Callable = preprocess_stats) -> pd.DataFrame:
    """
    Run and evaluate agent.

    :param data:        raw observations in pandas DataFrame
    :return:            result saves to the same path as input

    Args:
        preprocess: function to preprocess data
    """
    columns = ['Cell ID', 'LAC', 'HR Usage Rate', 'TCH Blocking Rate, BH', 'Number of Available\nTCH',
               'TCH Traffic (Erl), BH', 'Lower_limit', 'Upper_limit']

    df = preprocess(data, columns)
    obs_array = df.drop(columns=['Cell ID', 'LAC'], errors='ignore')
    obs_array.rename_axis(None, axis=1, inplace=True)
    obs_array.reset_index(drop=True, inplace=True)

    agent = load_agent('sac_last_60_50d_exp-r.pt', 'pt', )
    state_predictor = NHiTSModel.load_from_checkpoint("nhits_35lw_2l_1b_3s_35d_no_TB", "state_predictor", best=True)

    # # 'HR Usage Rate', 'TCH Blocking Rate, BH'
    # self.current_state = series[randint(0, len(series))].head(n_past)
    # # 'Number of Available\nTCH', 'TCH Traffic (Erl), BH', 'Param 1',  'Param 2'
    # self.cov = covariates[0].head(n_past)


    lower_limits = []
    upper_limits = []
    qualities = []
    new_states = []

    # 'HR Usage Rate', 'TCH Blocking Rate, BH'
    current_state = obs_array.iloc[:7, :2]
    cov = obs_array.iloc[:7, -4:]

    # print(TimeSeries.from_dataframe(obs_array.iloc[:, :2]))
    # print(len(TimeSeries.from_dataframe(obs_array.iloc[:, :2])))

    # setting env for reward calculation
    environment = SimulatedCustomEnv(
        state_predictor,
        np.array([1,1]),
        TimeSeries.from_dataframe(obs_array.iloc[:, :2]),
        TimeSeries.from_dataframe(obs_array.iloc[:, -4:]),
        7
    )
    obs = environment.reset()
    mom_reward = []

    for i, row in enumerate(obs_array.iloc[7:].values):
        # print('Curr_state=', current_state.shape)

        a1, a2 = predict(row, agent)
        lower = clip(int(row[-2] + a1 * 30))
        upper = clip(int(row[-1] + a2 * 30))

        # compure reward
        new_state, reward, done, info = environment.step(np.array([a1, a2]))
        mom_reward.append(reward)

        # Compute quality
        qualities.append(
            quality(blocking=row[1], ch=row[2], traffic=row[3], param1=row[-2], param2=row[-1], prparam1=lower,
                    prparam2=upper)
        )

        cov.iloc[-1, -2:] = (lower, upper)
        # print(cov)
        # n for number of states to predict
        # current_state.rename_axis(None, axis=1, inplace=True)
        # current_state.reset_index(drop=True, inplace=True)
        pred_state = state_predictor.predict(n=1, series=TimeSeries.from_dataframe(current_state),
                                             past_covariates=TimeSeries.from_dataframe(cov), verbose=False)
        new_states.append(pred_state)

        lower_limits.append(lower)
        upper_limits.append(upper)

        current_state = pd.concat([current_state.iloc[1:], obs_array.iloc[i +7: i+8, :2]], axis=0, join='inner')
        # print(current_state)

        cov = obs_array.iloc[i+1: i +8, -4:]
    # df['Lower_limit_Gen'], df['Upper_limit_Gen'], df['Limit_quality_Gen'] = lower_limits, upper_limits, qualities
    # df["Quality Rate"] = 1 - (2*df['HR Usage Rate']/100 + np.log(df['TCH Blocking Rate, BH'] + 1))/(1 + np.log(101))

    states_df = pd.concat(list(map(lambda x: x.pd_dataframe(), new_states)))
    states_df["Quality Rate"] = 1 - (2*states_df['HR Usage Rate']/100 + np.log(states_df['TCH Blocking Rate, BH'] + 1))/(1 + np.log(101))
    states_df['cum_reward'] = np.cumsum(mom_reward)
    states_df['mom_reward'] = mom_reward

    return states_df


def preprocess_full(data: pd.DataFrame, cols: List[str]=None):
    df = data.copy()
    cols = ['HR Usage Rate', 'TCH Blocking Rate, BH', 'Number of Available\nTCH',
               'TCH Traffic (Erl), BH', 'Lower_limit', 'Upper_limit']
    df.drop(columns='DATA', inplace=True)
    df.rename(columns={'Param 1': cols[-2], 'Param 2': cols[-1]}, inplace=True)
    return df[cols]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default='data/dataset_full.csv')
    parser.add_argument('--size', type=int, default=2)

    args = parser.parse_args()
    size = args.size

    df = pd.read_csv(args.path, index_col=0)

    cell_list = list(map(lambda x: x[0], df[['Cell ID']].value_counts().index[:10].tolist()))
    curr = df[df['Cell ID'].isin(cell_list)]
    reff = df[~df['Cell ID'].isin(cell_list)]

    scores = []

    import logging

    logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

        for cell in tqdm(df[['Cell ID']].value_counts().keys()[:size]):
            cell_data = df[df['Cell ID'] == cell]

            data_drift_report = Report(metrics=[
                DataDriftPreset(),
            ])
            data_drift_report.run(reference_data=reff, current_data=cell_data, )
            drift = data_drift_report.as_dict()['metrics'][0]['result']['share_of_drifted_columns']

            states = optimize_params(cell_data, preprocess=preprocess_full)

            scores.append({
                'cell_id': cell,
                'drift_score': drift,
                'quality_avg': states['Quality Rate'].mean(),
                'quality_min': states['Quality Rate'].min(),
                'quality_max': states['Quality Rate'].max(),
                'quality_std': states['Quality Rate'].std(),
                'cum_reward_avg': states['cum_reward'].mean(),
                'cum_reward_max': states['cum_reward'].max(),
                'cum_reward_std': states['cum_reward'].std(),
                'mom_reward_avg': states['mom_reward'].mean(),
                'mom_reward_min': states['mom_reward'].min(),
                'mom_reward_max': states['mom_reward'].max(),
                'mom_reward_std': states['mom_reward'].std(),
            })

    scores_df = pd.DataFrame(scores)
    scores_df.to_csv('drift_scores_rewards_300.csv')
