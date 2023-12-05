#!/usr/bin/env python
# coding: utf-8

# In[67]:


from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[3]:


train_results = pd.read_pickle('data/generated/drift/by_cell_agent/full_rewards_new_agent_train-train_no_sample20-23.pkl')
train_results


# In[13]:


for d_score in sorted(train_results.drift_score.unique()):
    print(d_score)


# In[18]:


train_results.drift_score.value_counts()


# In[22]:


test_results = pd.read_pickle('data/generated/drift/by_cell_agent/full_rewards_new_agent_train-test_no_sample20-23.pkl')
test_results


# In[66]:


test_results.drift_score.value_counts()


# In[30]:


type(train_results.drift_score)


# In[37]:


cols = train_results.columns[2:]
cols


# In[73]:


def reformat_full_results(df: pd.DataFrame) -> Dict[float, pd.DataFrame]:
    df_by_drift_results = {}

    for drift_score in sorted(df.drift_score.unique()):
        scores = df[df.drift_score == drift_score]

        for i in scores.index:
            series = scores.loc[i, cols]
            df_temp = pd.DataFrame()

            for s in series:
                df_temp = pd.concat([df_temp, s], axis=1)

            df_by_drift_results[drift_score] = pd.concat([df_by_drift_results.get(drift_score, pd.DataFrame()), df_temp])

    return df_by_drift_results


# In[74]:


train_by_drift_results = reformat_full_results(train_results)
test_by_drift_results = reformat_full_results(test_results)


# In[75]:


train_by_drift_results


# In[76]:


test_by_drift_results


# In[84]:


train_full = pd.concat(train_by_drift_results.values())
test_full = pd.concat(test_by_drift_results.values())
train_full


# In[86]:


fig = plt.figure(figsize =(10, 7))

# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])

# Creating plot
bp = ax.boxplot([train_full['Quality Rate'], test_full['Quality Rate']])
ax.set_xticklabels(['train', 'test'])
# show plot
plt.show()


# In[123]:


for col in train_full.columns:
    fig = plt.figure(figsize =(10, 7))

    # Creating axes instance
    ax = fig.add_axes([0, 0, 1, 1])

    # Creating plot
    bp = ax.boxplot([train_full[col], test_by_drift_results[0.875][col], test_by_drift_results[1.][col]],
                    patch_artist = True, notch ='True', vert = 0,  showfliers=False)
    ax.set_yticklabels(['train', 'test partial drift', 'test full drift'])
    print('Medians: ', [m.get_xdata()[0] for m in bp['medians']])
    # plt.xscale('log')
    plt.title(col)
    # show plot
    plt.show()


# In[121]:


for m in bp['medians']:
    print(m.get_xdata()[0])


# In[115]:


train_full.describe()


# In[90]:


test_full.describe()


# In[114]:


test_by_drift_results[1].describe()


# In[113]:


test_by_drift_results[0.875].describe()


# # Drift with on full dataset and Wasserstain.

# In[98]:


import json


def get_total_score(x: str) -> float:
    js = json.loads(x)
    size = len(js.items())

    return sum([js[f]['drift_detected'] for f,v in js.items()]) / size


# In[103]:


drift_scores_df = pd.read_csv('data/generated/drift/by_cell_agent/run_7/by_train_regressive__sampled-drift-None_no-window_.csv', index_col=0)
stat_tests = [c for c in drift_scores_df.columns if c not in ['cell_id']]

drift_scores_df['wasserstein_score'] = drift_scores_df[['wasserstein']].apply(lambda x: list(map(get_total_score, x)))

merged = test_results.merge(drift_scores_df, left_on='cell_id', right_on='cell_id', how='inner')
merged


# In[104]:


merged.wasserstein_score.value_counts()


# In[109]:


sum(test_results[test_results.drift_score == 1].cell_id == merged[merged.wasserstein_score == 1.].cell_id)


# In[ ]:




