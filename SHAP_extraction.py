#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import glob
import json

import numpy as np
import pandas as pd
# from stable_baselines3 import PPO
from tqdm import tqdm

import shap_extractor as se
from helpers import load_agent
from importlib import reload
from pathlib import Path

# from diploma.environment.plot import create_plots

reload(se)


# In[2]:


import pickle
with open("./data/series2.pkl", "rb") as f:
    series = pickle.load(f)

with open("./data/series23.pkl", "rb") as f:
    series23 = pickle.load(f)

with open("./data/covariates2.pkl", "rb") as f:
    cov = pickle.load(f)

with open("./data/covariates23.pkl", "rb") as f:
    cov23 = pickle.load(f)


# In[11]:


full_series=series + series23[100:]


# In[12]:


len(full_series)


# In[13]:


full_cov = cov+cov23[100:]


# In[7]:


full_cov


# In[16]:


list(map(lambda x: x.pd_dataframe(), full_series))


# In[17]:


df = pd.concat(list(map(lambda x: x.pd_dataframe(), full_series)))


# In[18]:


df_cov = pd.concat(list(map(lambda x: x.pd_dataframe(), full_cov)))


# In[29]:


full_df = pd.concat([df, df_cov], axis=1)


# In[30]:


full_df.columns


# In[31]:


full_df.reset_index(drop=True, inplace=True)
full_df.rename_axis(None, axis=1, inplace=True)
full_df.columns


# In[32]:


full_df


# In[33]:


full_df.to_csv("./data/new_full_param_data.csv")


# # Cell agent

# In[37]:


# import Bachelor.rl as rl

data_path = 'Bachelor/data/dataset_full.csv'
agent = load_agent('sac_best_enough_qual.pt', 'pt',)


# In[34]:


import torch


def preprocess(data: pd.DataFrame):
    df = data.copy()
    cols = ['HR Usage Rate', 'TCH Blocking Rate, BH', 'Number of Available\nTCH',
               'TCH Traffic (Erl), BH', 'Lower_limit', 'Upper_limit']
    df.drop(columns='DATA', inplace=True)
    df.rename(columns={'Param 1': cols[-2], 'Param 2': cols[-1]}, inplace=True)

    df = df[cols].drop(columns=['Cell ID', 'LAC'], errors='ignore')
    df.rename_axis(None, axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df[cols]

def cell_f(x):
    res = agent.act(torch.as_tensor(x, dtype=torch.float32))

    return res



# In[12]:


data = pd.read_csv(data_path, index_col=0)


# In[13]:


data


# In[28]:


# from tqdm import tqdm
# 
# expected_values = []
# path = 'data/cell/run_1/'
# 
# for cell in tqdm(data['Cell ID'].unique()):
#     cell_df = data[data['Cell ID'] == cell]
#     cell_df = preprocess(cell_df)
# 
#     path, exp_value = se.extract_shap(cell_df, cell_f, path, f'cell_{cell}')
#     expected_values.append((cell, exp_value))
# 
# exp_df = pd.DataFrame(expected_values, columns=['cell_id', 'expected_value'])
# exp_df.to_csv(os.path.join(path, 'expected_values.csv'))


# In[35]:


full_df


# In[38]:


from tqdm import tqdm

expected_values = []
path = 'data/cell/run_1/'

cell_df = full_df.copy()

path, exp_value = se.extract_shap(cell_df, cell_f, path, f'cell_all_train')
expected_values.append(('all_train', exp_value))

exp_df = pd.DataFrame(expected_values, columns=['cell_id', 'expected_value'])
exp_df.to_csv(os.path.join(path, 'expected_values.csv'))


# In[24]:


np.ones((2, 10, 6))



# In[ ]:





# ## 2020-2023 agent on 994 cells

# In[4]:


import Bachelor.rl as rl

data_path = 'data/train_2020-2023.csv'
sys.path.append("Bachelor/")
agent = load_agent('sac_best_new_alpha4.pt', 'pt', 'agent')


# In[5]:


import torch


def preprocess(data: pd.DataFrame):
    df = data.copy()
    cols = ['HR Usage Rate', 'TCH Blocking Rate, BH', 'Number of Available\nTCH',
               'TCH Traffic (Erl), BH', 'Lower_limit', 'Upper_limit']
    df.drop(columns='DATA', inplace=True)
    df.rename(columns={'Param 1': cols[-2], 'Param 2': cols[-1]}, inplace=True)

    df = df[cols].drop(columns=['Cell ID', 'LAC'], errors='ignore')
    df.rename_axis(None, axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df[cols]

def cell_f(x):
    res = agent.act(torch.as_tensor(x, dtype=torch.float32))

    return res



# In[8]:


data = pd.read_csv(data_path, )


# In[9]:


data


# In[10]:


from tqdm import tqdm

expected_values = []
path = 'data/shap/agent_2020-2023/'
Path(path).mkdir(parents=True, exist_ok=True)

for cell in tqdm(data['Cell ID'].unique()):
    cell_df = data[data['Cell ID'] == cell]
    cell_df = preprocess(cell_df)

    path, exp_value = se.extract_shap(cell_df, cell_f, path, f'cell_{cell}')
    expected_values.append((cell, exp_value))

exp_df = pd.DataFrame(expected_values, columns=['cell_id', 'expected_value'])
exp_df.to_csv(os.path.join(path, 'expected_values.csv'))


# In[25]:


pd.DataFrame(columns=cell_df.columns)


# In[27]:


shap_df = pd.DataFrame(columns=cell_df.columns)

for cell_path in sorted(glob.glob(f'{path}/*.pkl')):
    tmp = np.array(pd.read_pickle(cell_path)).__abs__().sum(0)
    tmp = pd.DataFrame(tmp, columns=shap_df.columns)
    shap_df = pd.concat([shap_df, tmp])

shap_df


# In[31]:


fi = pd.DataFrame(shap_df.sum(0), columns=['importance'])
fi


# In[32]:


fi.to_csv('data/generated/fi_2020-2023.csv')


# In[ ]:





# In[ ]:





# # Trading

# In[2]:


path_to_json = 'diploma/experiments/16_05_new_reward_window_size_16_ups_False/stop_loss_ne_100_rs_True_eer_False_ffh_0/profits_train/'
list_of_jsons = glob.glob(path_to_json + '*.json')
list_of_jsons = sorted(list_of_jsons)

for json_file in list_of_jsons:
    profits_dict = json.load(open(json_file, "r"))
    print(profits_dict)
    print()
    # break


# In[3]:


infile = '/media/rid/Files/Datasets/Magister/2021_year/'
use_predefined_scaler = False


# In[4]:


list_of_weeks = [f'{infile}week_{i}.parquet' for i in range(10, 20, 1)]


# In[5]:


list_of_weeks


# In[8]:


from diploma.environment.crypto_env_random_price_new_reward import CryptoEnv

if use_predefined_scaler:
    train_env = False
    if len(list_of_weeks) == 4:
        path_to_scaler = f"{infile}data_scalers_4_days/"
    elif len(list_of_weeks) == 6:
        path_to_scaler = f"{infile}data_scalers_6_days/"
    elif len(list_of_weeks) == 10:
        path_to_scaler = f"{infile}data_scalers_10_days/"
    elif len(list_of_weeks) == 15:
        path_to_scaler = f"{infile}data_scalers_15_days/"
else:
    train_env = True
    path_to_scaler = "diploma/all_scalers/scalers_4/"
    Path(path_to_scaler).mkdir(parents=True, exist_ok=True)

config = {
    'transaction_cost': 0.0001, #0.01
    'window_size': 16,
    'n_shares': 1,
    'n_epochs': 100,
    'train_env': train_env,
    'random_start': True,
    'df_path': infile,
    'path_to_scaler': f"{infile}data_scalers\\",
    'scaler_to_use': 'QuantileTransformer',
    'initial_investment': 100000,
    'end_episode_reward': False,
    'add_metadata': True,
    'fine_for_holding': 0
}

random_start = config['random_start']
n_epochs_for_tune = 10

config.update({'df_path': list_of_weeks[0],
               'train_env': train_env,
               'random_start': random_start})

env = CryptoEnv(config)

model_path = 'diploma/experiments/16_05_new_reward_window_size_16_ups_False/stop_loss_ne_100_rs_True_eer_False_ffh_0/ppo_models/ppo_model_trained_on_week_14.zip'
model = PPO.load(model_path, env=env)

env.close()


# In[10]:


def trading_f(x):
    act, _ = model.predict(x)

    return act


# In[ ]:


profits = []
expected_values = []
profits_path = 'data/trading/run_1/'
Path(profits_path).mkdir(parents=True, exist_ok=True)
path = profits_path

for idw_test, week_test in enumerate(list_of_weeks[-5:]):
    config.update({'df_path': week_test,
                   'train_env': True,
                   'random_start': False,
                   'initial_investment': 70_000,})

    env = CryptoEnv(config)

    week_test_name = str(Path(week_test.replace('\\', '/')).name)[:7]

    net_json = {0: [],
                1: [],
                2: [],
                3: [],
                4: [],
                5: [],
                6: [],}
    net = []
    actions = []
    positions = []
    share_prices = []
    observations = []

    obs = env.reset()
    for i in tqdm(range(len(env.states) - 2)):
        action, _states = model.predict(obs)
        actions.append(action)
        observations.append(obs)

        obs, rewards, done, info = env.step(action)
        positions.append(env.position)
        share_prices.append(env.current_price)

        current_state_index, balance, own_share, net_worth, profit = env.render()
        net_json[i // 1440].append(net_worth)
        net.append(net_worth)

        # if net_worth - config['initial_investment'] < -2000:
        #     break

    # Close the processes
    env.close()

    for i in range(7):
        profit = net_json[i][-1] - net_json[i][0]
        profits.append((week_test, i, profit))
    #get shap
    week_name = f"week_{week_test[week_test.rfind('week_') + 5: week_test.rfind('.parquet')]}"
    path, exp_value = se.extract_shap(pd.DataFrame(observations), trading_f, path, week_name)
    expected_values.append((week_name, exp_value))

    # fig = create_plots(net, positions, actions, share_prices)
    # fig.savefig(f'{plots_path}results_{week_test_name}.jpg', dpi=300)

exp_df = pd.DataFrame(expected_values, columns=['week', 'expected_value'])
exp_df.to_csv(os.path.join(path, 'expected_values.csv'))

with open(profits_path + f'profits.json', 'w') as f:
    f.write(json.dumps(profits))


# In[ ]:




