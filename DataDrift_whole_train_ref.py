#!/usr/bin/env python
# coding: utf-8

# In[3]:


import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Callable, List, Optional, Dict, Union

from evidently.test_preset import DataDriftTestPreset
from evidently.test_suite import TestSuite
from tqdm import tqdm

from helpers import predict, load_agent, quality, clip
from preprocess import preprocess_stats
from rl.sim_enviroment import SimulatedCustomEnv

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.options import DataDriftOptions
from drift_calculator import DriftCalculator


# In[4]:


train_df = pd.read_csv('data/train_2020-2023.csv', )
test_df = pd.read_csv('data/test_2020-2023.csv', )


# In[5]:


train_df


# In[6]:


test_df


# In[6]:


test_df.index


# In[7]:


len(train_df['Cell ID'].unique()), len(set(train_df['Cell ID'].unique()).intersection(test_df['Cell ID'].unique()))


# In[7]:


stat_tests = [
    'ks', # <= 1000 Kolmogorov–Smirnov
    'wasserstein', # > 1000 Wasserstein distance (normed)
    'kl_div', # Kullback-Leibler divergence
    'psi', # Population Stability Index
    'jensenshannon',  #  > 1000 Jensen-Shannon distance
    # 'anderson', # Anderson-Darling test
    'cramer_von_mises', # Cramer-Von-Mises test
    'hellinger', # Hellinger Distance (normed)
    'mannw', # Mann-Whitney U-rank test
    'ed', # Energy distance
    # 'es', # Epps-Singleton tes
    't_test', # T-Test
    'emperical_mmd', # Emperical-MMD
]


# In[38]:


save_path = 'data/generated/drift/by_cell_agent/run_1/'
Path(save_path).mkdir(exist_ok=True, parents=True)

drift_calc = DriftCalculator([None] + stat_tests, ['default'] + stat_tests)
drift_scores = []
cols = ['Number of Available\nTCH', 'HR Usage Rate', 'TCH Blocking Rate, BH',
       'TCH Traffic (Erl), BH', 'Param 1', 'Param 2']
ref = train_df

for cell in tqdm(train_df['Cell ID'].value_counts().keys()[:]):
    # ref = train_df[train_df['Cell ID'] == cell]
    cur = test_df[test_df['Cell ID'] == cell]
    # add original distribution
    # cur = pd.concat([ref.sample(n=len(ref) - len(cur)), cur])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = drift_calc.get_drift(current_data=cur, reference_data=ref, sample=True, weighted=False, save_plot=f'{save_path}cell_{cell}')

    score['cell_id'] = cell
    drift_scores.append(score)

drift_scores_df = pd.DataFrame(drift_scores)
drift_scores_df.to_csv(f'{save_path}/by_cell_sampled.csv')
drift_scores_df


# In[39]:


drift_scores_df.describe()


# In[40]:


rewards = pd.read_csv('data/generated/drift/by_cell_agent/drift_scores_rewards_new_agent_train-test_no_sample20-23.csv', index_col=0)
rewards


# In[41]:


merged = rewards.merge(drift_scores_df, left_on='cell_id', right_on='cell_id', how='inner')
merged


# In[42]:


merged.corr()


# In[37]:


merged[merged['wasserstein'] != merged['drift_score']]


# ## No sample | evidently 0.4.5

# In[33]:


stat_tests = [
    'ks', # <= 1000 Kolmogorov–Smirnov
    'wasserstein', # > 1000 Wasserstein distance (normed)
    'kl_div', # Kullback-Leibler divergence
    'psi', # Population Stability Index
    'jensenshannon',  #  > 1000 Jensen-Shannon distance
    # # 'anderson', # Anderson-Darling test
    'cramer_von_mises', # Cramer-Von-Mises test
    'hellinger', # Hellinger Distance (normed)
    # 'mannw', # Mann-Whitney U-rank test (too long ~23 s on iteration)
    'ed', # Energy distance
    # # 'es', # Epps-Singleton tes
    't_test', # T-Test
    # 'emperical_mmd', # Emperical-MMD (takes too much space to compute on sampled data)
]


# In[34]:


save_path = 'data/generated/drift/by_cell_agent/run_2/'
Path(save_path).mkdir(exist_ok=True, parents=True)

drift_calc = DriftCalculator([None] + stat_tests , ['default'] + stat_tests )
drift_scores = []
cols = ['Number of Available\nTCH', 'HR Usage Rate', 'TCH Blocking Rate, BH',
       'TCH Traffic (Erl), BH', 'Param 1', 'Param 2']
ref = train_df

for cell in tqdm(train_df['Cell ID'].value_counts().keys()[:]):
    # ref = train_df[train_df['Cell ID'] == cell]
    cur = test_df[test_df['Cell ID'] == cell]
    # add original distribution
    # cur = pd.concat([ref.sample(n=len(ref) - len(cur)), cur])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = drift_calc.get_drift(current_data=cur, reference_data=ref, sample=False, weighted=False, save_plot=None)
        # data_drift_report = Report(metrics=[
        #     DataDriftPreset(),
        # ])
        # data_drift_report.run(reference_data=ref, current_data=cur,)
        # drift = data_drift_report.as_dict()['metrics'][0]['result']['share_of_drifted_columns']
        # score = {'default': drift}

    score['cell_id'] = cell
    drift_scores.append(score)

drift_scores_df = pd.DataFrame(drift_scores)
drift_scores_df.to_csv(f'{save_path}/by_cell_not-sampled.csv')
drift_scores_df


# In[32]:


import time

for t in stat_tests:
    start_time = time.time()

    data_drift_report = Report(metrics=[
        DataDriftPreset(num_stattest=t),
    ])
    data_drift_report.run(reference_data=ref, current_data=cur,)
    drift = data_drift_report.as_dict()['metrics'][0]['result']['share_of_drifted_columns']

    print(f'{t} : {(time.time() - start_time) } s')


# In[14]:


rewards = pd.read_csv('data/generated/drift/by_cell_agent/drift_scores_rewards_new_agent_train-test_no_sample20-23.csv', index_col=0)
rewards


# In[15]:


merged = rewards.merge(drift_scores_df, left_on='cell_id', right_on='cell_id', how='inner')
merged


# In[16]:


merged.corr()


# In[ ]:




