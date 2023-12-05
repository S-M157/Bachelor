#!/usr/bin/env python
# coding: utf-8

# In[202]:


from functools import partial
from pathlib import Path
from typing import List, Dict, Callable

import pandas as pd
import numpy as np
import json


# In[204]:


from functools import wraps, WRAPPER_ASSIGNMENTS

try:
    wraps(partial(wraps))(wraps)
except AttributeError:
    @wraps(wraps)
    def wraps(obj, attr_names=WRAPPER_ASSIGNMENTS, wraps=wraps):
        return wraps(obj, assigned=(name for name in attr_names if hasattr(obj, name)))


# In[3]:


def get_total_score(x: str) -> float:
    js = json.loads(x)
    size = len(js.items())

    return sum([js[f]['drift_detected'] for f,v in js.items()]) / size

def get_regress_score(x: str) -> float:
    js = json.loads(x)

    # TODO: use different aggregation methods
    return sum([js[f]['drift_score'] for f,v in js.items()])


# In[78]:


def get_weighted_aggregation(x: str, fi: Dict, method: Callable = sum):
    js = json.loads(x)

    return method([js[f]['drift_score'] * fi.get(f, 1) for f,v in js.items()])


# In[87]:


def get_weighted_thresh_agg(x: str, fi: Dict, method: Callable = sum):
    js = json.loads(x)

    return method([js[f]['drift_score'] * fi.get(f, 1) // js[f]['stattest_threshold'] for f,v in js.items()])


# In[220]:


fi = pd.read_csv('data/generated/fi_2020-2023.csv', index_col=0)
fi.rename(index={'Lower_limit': 'Param 1', 'Upper_limit': 'Param 2'}, inplace=True)
fi = fi / fi.max()
fi = fi.to_dict()['importance']
fi


# In[229]:


get_weighted_thresh_sum = partial(get_weighted_thresh_agg, fi=fi, method=np.average)
get_weighted_thresh_sum.__name__ = 'get_weighted_thresh_sum'

get_thresh_sum = partial(get_weighted_thresh_agg, fi={}, method=np.average)
get_thresh_sum.__name__ = 'get_thresh_sum'

get_weighted_mean = partial(get_weighted_aggregation, fi=fi, method=np.sum)
get_weighted_mean.__name__ = 'get_weighted_mean'


# In[216]:


get_weighted_thresh_sum.__name__


# In[4]:


rewards = pd.read_csv('data/generated/drift/by_cell_agent/drift_scores_rewards_new_agent_train-test_no_sample20-23.csv', index_col=0)
rewards


# In[154]:


rewards.describe()


# In[434]:


rows = [c for c in rewards.columns if c not in ['cell_id', 'drift_score']]


# # NOT Sampled | no ref window

# In[435]:


orig_df = pd.read_csv('data/generated/drift/by_cell_agent/run_7/by_train_regressive__sampled-drift-None_no-window_.csv', index_col=0)
orig_df


# In[436]:


stat_tests = [c for c in orig_df.columns if c not in ['cell_id']]
cols = [c for c in orig_df.columns if c not in rows + ['cell_id', 'drift_score']]

drift_scores_df = orig_df[['cell_id']].copy()

drift_scores_df[stat_tests] = orig_df[stat_tests].apply(lambda x: list(map(get_total_score, x)))
drift_scores_df


# In[450]:


merged = rewards.merge(drift_scores_df, left_on='cell_id', right_on='cell_id', how='inner')
merged.corr().loc[rows, cols].style.background_gradient(cmap ='coolwarm', vmin=-1, vmax=1, axis=1)


# ## Regress

# In[232]:


drift_regress_df = orig_df[['cell_id']].copy()

drift_regress_df[stat_tests] = orig_df[stat_tests].apply(lambda x: list(map(get_regress_score, x)))
drift_regress_df


# In[451]:


merged = rewards.merge(drift_regress_df, left_on='cell_id', right_on='cell_id', how='inner')
merged.corr().loc[rows, cols].style.background_gradient(cmap ='coolwarm', vmin=-1, vmax=1, axis=1)


# In[452]:


merged.corr(method='kendall').loc[rows, cols].style.background_gradient(cmap ='coolwarm', vmin=-1, vmax=1, axis=1)


# In[448]:


merged.corr(method='spearman').loc[rows, cols].style.background_gradient(cmap ='coolwarm', vmin=-1, vmax=1, axis=1)


# ## Weighted

# In[453]:


df_weighted = orig_df[['cell_id']].copy()

df_weighted[stat_tests] = orig_df[stat_tests].apply(lambda x: list(map(get_weighted_mean, x)))
df_weighted


# In[454]:


merged = rewards.merge(df_weighted, left_on='cell_id', right_on='cell_id', how='inner')
merged.corr().loc[rows, cols].style.background_gradient(cmap ='coolwarm', vmin=-1, vmax=1, axis=1)


# ## Adaptive threshold

# In[455]:


df_thresh = orig_df[['cell_id']].copy()

df_thresh[stat_tests] = orig_df[stat_tests].apply(lambda x: list(map(get_thresh_sum, x)))
df_thresh


# In[456]:


merged_thresh = rewards.merge(df_thresh, left_on='cell_id', right_on='cell_id', how='inner')
merged_thresh.corr().loc[rows, cols].style.background_gradient(cmap ='coolwarm', vmin=-1, vmax=1, axis=1)


# ## Adaptive weighted threshold

# In[457]:


df_weighted_thresh = orig_df[['cell_id']].copy()

df_weighted_thresh[stat_tests] = orig_df[stat_tests].apply(lambda x: list(map(get_weighted_thresh_sum, x)))
df_weighted_thresh


# In[458]:


df_weighted_thresh = rewards.merge(df_weighted_thresh, left_on='cell_id', right_on='cell_id', how='inner')
df_weighted_thresh.corr().loc[rows, cols].style.background_gradient(cmap ='coolwarm', vmin=-1, vmax=1, axis=1)


# # Sampled | no ref window

# In[459]:


orig_df = pd.read_csv('data/generated/drift/by_cell_agent/run_8/by_train_regressive__sampled-drift-1000_no-window_.csv', index_col=0)
orig_df


# In[460]:


stat_tests = [c for c in orig_df.columns if c not in ['cell_id']]

drift_scores_df = orig_df[['cell_id']].copy()

drift_scores_df[stat_tests] = orig_df[stat_tests].apply(lambda x: list(map(get_total_score, x)))
drift_scores_df


# ## Regress

# In[461]:


drift_regress_df = orig_df[['cell_id']].copy()

drift_regress_df[stat_tests] = orig_df[stat_tests].apply(lambda x: list(map(get_regress_score, x)))
drift_regress_df


# In[462]:


merged = rewards.merge(drift_scores_df, left_on='cell_id', right_on='cell_id', how='inner')
merged.corr().loc[rows, cols].style.background_gradient(cmap ='coolwarm', vmin=-1, vmax=1, axis=1)


# In[463]:


merged = rewards.merge(drift_regress_df, left_on='cell_id', right_on='cell_id', how='inner')
merged.corr().loc[rows, cols].style.background_gradient(cmap ='coolwarm', vmin=-1, vmax=1, axis=1)


# In[464]:


merged.corr(method='kendall').loc[rows, cols].style.background_gradient(cmap ='coolwarm', vmin=-1, vmax=1, axis=1)


# In[465]:


merged.corr(method='spearman').loc[rows, cols].style.background_gradient(cmap ='coolwarm', vmin=-1, vmax=1, axis=1)


# ## Weighted

# In[466]:


df_weighted = orig_df[['cell_id']].copy()

df_weighted[stat_tests] = orig_df[stat_tests].apply(lambda x: list(map(get_weighted_mean, x)))
df_weighted


# In[467]:


merged = rewards.merge(df_weighted, left_on='cell_id', right_on='cell_id', how='inner')
merged.corr().loc[rows, cols].style.background_gradient(cmap ='coolwarm', vmin=-1, vmax=1, axis=1)


# ## Adaptive threshold

# In[468]:


df_thresh = orig_df[['cell_id']].copy()

df_thresh[stat_tests] = orig_df[stat_tests].apply(lambda x: list(map(get_thresh_sum, x)))
df_thresh


# In[469]:


merged_thresh = rewards.merge(df_thresh, left_on='cell_id', right_on='cell_id', how='inner')
merged_thresh.corr().loc[rows, cols].style.background_gradient(cmap ='coolwarm', vmin=-1, vmax=1, axis=1)


# ## Adaptive weighted threshold

# In[470]:


df_weighted_thresh = orig_df[['cell_id']].copy()

df_weighted_thresh[stat_tests] = orig_df[stat_tests].apply(lambda x: list(map(get_weighted_thresh_sum, x)))
df_weighted_thresh


# In[471]:


df_weighted_thresh = rewards.merge(df_weighted_thresh, left_on='cell_id', right_on='cell_id', how='inner')
df_weighted_thresh.corr().loc[rows, cols].style.background_gradient(cmap ='coolwarm', vmin=-1, vmax=1, axis=1)


# # NOT Sampled | ref window 1000

# In[472]:


orig_df = pd.read_csv('data/generated/drift/by_cell_agent/run_9/by_train_regressive_sampled_ref_sampled_drift_None_window_1k_.csv', index_col=0)
orig_df


# In[473]:


stat_tests = [c for c in orig_df.columns if c not in ['cell_id']]

drift_scores_df = orig_df[['cell_id']].copy()

drift_scores_df[stat_tests] = orig_df[stat_tests].apply(lambda x: list(map(get_total_score, x)))
drift_scores_df


# In[474]:


merged = rewards.merge(drift_scores_df, left_on='cell_id', right_on='cell_id', how='inner')
merged.corr().loc[rows, cols].style.background_gradient(cmap ='coolwarm', vmin=-1, vmax=1, axis=1)


# ## Regress

# In[475]:


drift_regress_df = orig_df[['cell_id']].copy()

drift_regress_df[stat_tests] = orig_df[stat_tests].apply(lambda x: list(map(get_regress_score, x)))
drift_regress_df


# In[476]:


merged = rewards.merge(drift_regress_df, left_on='cell_id', right_on='cell_id', how='inner')
merged.corr().loc[rows, cols].style.background_gradient(cmap ='coolwarm', vmin=-1, vmax=1, axis=1)


# In[477]:


merged.corr(method='kendall').loc[rows, cols].style.background_gradient(cmap ='coolwarm', vmin=-1, vmax=1, axis=1)


# In[478]:


merged.corr(method='spearman').loc[rows, cols].style.background_gradient(cmap ='coolwarm', vmin=-1, vmax=1, axis=1)


# ## Weighted

# In[479]:


df_weighted = orig_df[['cell_id']].copy()

df_weighted[stat_tests] = orig_df[stat_tests].apply(lambda x: list(map(get_weighted_mean, x)))
df_weighted


# In[480]:


merged = rewards.merge(df_weighted, left_on='cell_id', right_on='cell_id', how='inner')
merged.corr().loc[rows, cols].style.background_gradient(cmap ='coolwarm', vmin=-1, vmax=1, axis=1)


# ## Adaptive threshold

# In[481]:


df_thresh = orig_df[['cell_id']].copy()

df_thresh[stat_tests] = orig_df[stat_tests].apply(lambda x: list(map(get_thresh_sum, x)))
df_thresh


# In[482]:


merged_thresh = rewards.merge(df_thresh, left_on='cell_id', right_on='cell_id', how='inner')
merged_thresh.corr().loc[rows, cols].style.background_gradient(cmap ='coolwarm', vmin=-1, vmax=1, axis=1)


# ## Adaptive weighted threshold

# In[483]:


df_weighted_thresh = orig_df[['cell_id']].copy()

df_weighted_thresh[stat_tests] = orig_df[stat_tests].apply(lambda x: list(map(get_weighted_thresh_sum, x)))
df_weighted_thresh


# In[484]:


df_weighted_thresh = rewards.merge(df_weighted_thresh, left_on='cell_id', right_on='cell_id', how='inner')
df_weighted_thresh.corr().loc[rows, cols].style.background_gradient(cmap ='coolwarm', vmin=-1, vmax=1, axis=1)


# # Sampled | ref window 1000

# In[644]:


orig_df = pd.read_csv('data/generated/drift/by_cell_agent/run_10/by_train_regressive_sampled_ref_sampled_drift_1000_window_1k_.csv', index_col=0)
orig_df


# In[645]:


stat_tests = [c for c in orig_df.columns if c not in ['cell_id']]

drift_scores_df = orig_df[['cell_id']].copy()

drift_scores_df[stat_tests] = orig_df[stat_tests].apply(lambda x: list(map(get_total_score, x)))
drift_scores_df


# In[646]:


merged = rewards.merge(drift_scores_df, left_on='cell_id', right_on='cell_id', how='inner')
merged.corr().loc[rows, cols].style.background_gradient(cmap ='coolwarm', vmin=-1, vmax=1, axis=1)


# ## Regress

# In[498]:


drift_regress_df = orig_df[['cell_id']].copy()

drift_regress_df[stat_tests] = orig_df[stat_tests].apply(lambda x: list(map(get_regress_score, x)))
drift_regress_df


# In[499]:


merged = rewards.merge(drift_regress_df, left_on='cell_id', right_on='cell_id', how='inner')
merged.corr().loc[rows, cols].style.background_gradient(cmap ='coolwarm', vmin=-1, vmax=1, axis=1)


# In[500]:


merged.corr(method='kendall').loc[rows, cols].style.background_gradient(cmap ='coolwarm', vmin=-1, vmax=1, axis=1)


# In[501]:


merged.corr(method='spearman').loc[rows, cols].style.background_gradient(cmap ='coolwarm', vmin=-1, vmax=1, axis=1)


# ## Weighted

# In[502]:


df_weighted = orig_df[['cell_id']].copy()

df_weighted[stat_tests] = orig_df[stat_tests].apply(lambda x: list(map(get_weighted_mean, x)))
df_weighted


# In[503]:


merged = rewards.merge(df_weighted, left_on='cell_id', right_on='cell_id', how='inner')
merged.corr().loc[rows, cols].style.background_gradient(cmap ='coolwarm', vmin=-1, vmax=1, axis=1)


# ## Other funcs

# In[504]:


orig_df


# In[505]:


df_weighted = orig_df[['cell_id']].copy()

df_weighted[stat_tests] = orig_df[stat_tests].apply(lambda x: list(map(get_weighted_thresh_sum, x)))
rewards.merge(df_weighted, left_on='cell_id', right_on='cell_id', how='inner').corr()


# In[506]:


df_weighted


# ## Adaptive threshold

# In[507]:


df_thresh = orig_df[['cell_id']].copy()

df_thresh[stat_tests] = orig_df[stat_tests].apply(lambda x: list(map(get_thresh_sum, x)))
df_thresh


# In[508]:


merged_thresh = rewards.merge(df_thresh, left_on='cell_id', right_on='cell_id', how='inner')
merged_thresh.corr().loc[rows, cols].style.background_gradient(cmap ='coolwarm', vmin=-1, vmax=1, axis=1)


# ## Adaptive weighted threshold

# In[509]:


df_weighted_thresh = orig_df[['cell_id']].copy()

df_weighted_thresh[stat_tests] = orig_df[stat_tests].apply(lambda x: list(map(get_weighted_thresh_sum, x)))
df_weighted_thresh


# In[510]:


df_weighted_thresh = rewards.merge(df_weighted_thresh, left_on='cell_id', right_on='cell_id', how='inner')
df_weighted_thresh.corr().loc[rows, cols].style.background_gradient(cmap ='coolwarm', vmin=-1, vmax=1, axis=1)


# # Get data for learning

# In[185]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from joblib import dump, load
import os.path as osp
import numpy as np


# In[153]:


def get_by_feature_score(x: str) -> Dict[str, any]:
    js = json.loads(x)

    return {f: js[f]['drift_score'] for f,v in js.items()}


# In[368]:


def get_weighted_poly_aggregation(x: str, fi: Dict, method: Callable = lambda x: x) -> Dict:
    js = json.loads(x)

    return method({f: js[f]['drift_score'] * fi.get(f, 1) for f,v in js.items()})


# In[369]:


def get_weighted_poly_thresh_agg(x: str, fi: Dict, method: Callable = lambda x: x) -> Dict:
    js = json.loads(x)

    return method({f: js[f]['drift_score'] * fi.get(f, 1) // js[f]['stattest_threshold'] for f,v in js.items()})


# In[370]:


get_weighted_poly = partial(get_weighted_poly_aggregation, fi=fi, method=lambda x: x)
get_weighted_poly.__name__ = 'get_weighted_poly'

get_weighted_thresh_poly = partial(get_weighted_poly_thresh_agg, fi=fi, method=lambda x: x)
get_weighted_thresh_poly.__name__ = 'get_weighted_thresh_poly'

get_thresh_poly = partial(get_weighted_poly_thresh_agg, fi={}, method=lambda x: x)
get_thresh_poly.__name__ = 'get_thresh_poly'


# In[371]:


def get_models_results(df: pd.DataFrame,
                       rewards_df: pd.DataFrame,
                       features_l: List[str],
                       target: str,
                       stat_ts: List[str],
                       save_path: str,
                       data_func: Callable) -> pd.DataFrame:
    df_4_train = df[['cell_id']].copy()
    df_4_train[stat_ts] = df[stat_ts].apply(lambda x: list(map(data_func, x)))

    results = []

    for st in stat_ts:
        train_df = pd.DataFrame(df_4_train[st].tolist(), index=df_4_train['cell_id'])
        train_df = train_df.merge(rewards_df[['cell_id', target]], left_on='cell_id', right_on='cell_id', how='inner')

        X_train, X_test, y_train, y_test = train_test_split(train_df[features_l], train_df[target], test_size=0.3, random_state=3407)

        # Linear Regression
        model_lr = LinearRegression()
        model_lr.fit(X_train, y_train)
        lr_train_score = model_lr.score(X_train, y_train)
        lr_test_score = model_lr.score(X_test, y_test)
        # dumping model
        lr_path = osp.join(save_path, f'lr_{st}.joblib')
        dump(model_lr, lr_path)

        # Polynomial
        transformer = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_train_polynom = transformer.fit_transform(X_train)
        X_test_polynom = transformer.transform(X_test)
        model_pol_lr = LinearRegression().fit(X_train_polynom, y_train)
        pr_train_score = model_pol_lr.score(X_train_polynom, y_train)
        pr_test_score = model_pol_lr.score(X_test_polynom, y_test)
        # dumping model
        pol_lr_path = osp.join(save_path, f'pol_lr_{st}.joblib')
        trans_path = osp.join(save_path, f'transformer_{st}.joblib')
        dump(model_pol_lr, pol_lr_path)
        dump(transformer, trans_path)


        # save results
        results.append((st,
                        lr_train_score, lr_test_score,
                        pr_train_score, pr_test_score,
                        lr_path,
                        pol_lr_path,
                        trans_path))

    res_df = pd.DataFrame(results, columns=['stat_test',
                                            'lr_train_score', 'lr_test_score',
                                            'pol_lr_train_score', 'pol_lr_test_score',
                                            'lr_path',
                                            'pol_lr_path',
                                            'trans_path'])

    return res_df


# In[159]:


features = list(fi.keys())
features


# In[160]:


rewards.columns


# In[166]:


label = 'quality_avg'


# In[372]:


path = 'data/generated/models/one_drift_0'
get_models_results(orig_df, rewards, features, label, stat_tests, path, get_by_feature_score)


# ## Getting one drift all run

# In[199]:


fi


# In[200]:


stat_tests


# In[373]:


datasets_list = [('not-sampled_no-ref-window', 'data/generated/drift/by_cell_agent/run_7/by_train_regressive__sampled-drift-None_no-window_.csv'),
             ('sampled_no-ref-window', 'data/generated/drift/by_cell_agent/run_8/by_train_regressive__sampled-drift-1000_no-window_.csv'),
             ('not-sampled_ref-window-1k', 'data/generated/drift/by_cell_agent/run_9/by_train_regressive_sampled_ref_sampled_drift_None_window_1k_.csv'),
             ('sampled_ref-window-1k', 'data/generated/drift/by_cell_agent/run_10/by_train_regressive_sampled_ref_sampled_drift_1000_window_1k_.csv')]
path = 'data/generated/models/one_drift_1'
label = 'quality_avg'
features = ['HR Usage Rate',
 'TCH Blocking Rate, BH',
 'Number of Available\nTCH',
 'TCH Traffic (Erl), BH',
 'Param 1',
 'Param 2']
one_drift_res = pd.DataFrame()

for dataset_name, dataset_path in datasets_list:
    dataset = pd.read_csv(dataset_path, index_col=0)

    for form_data_func in [get_by_feature_score, get_weighted_poly, get_thresh_poly, get_weighted_thresh_poly]:
        data_type_name = form_data_func.__name__

        # regress_df = dataset[['cell_id']].copy()
        # regress_df[stat_tests] = dataset[stat_tests].apply(lambda x: list(map(form_data_func, x)))
        # creating path to save data
        saving_path = osp.join(path, dataset_name, data_type_name)
        Path(saving_path).mkdir(exist_ok=True, parents=True)

        res = get_models_results(dataset, rewards, features, label, stat_tests, saving_path, form_data_func)
        res['dataset_name'] = [dataset_name] * len(res)
        res['data_type_name'] = [data_type_name] * len(res)

        one_drift_res = pd.concat([one_drift_res, res])

one_drift_res.to_csv(osp.join(path, 'one_drift_res.csv'))
one_drift_res


# In[513]:


one_drift_res.groupby(by=['dataset_name', 'data_type_name'], ).max()


# In[421]:


ind_cols = ['dataset_name', 'data_type_name']
not_ind_cols = [c for c in one_drift_res.columns if c not in ind_cols]
one_drift_res_agg = pd.DataFrame(columns=not_ind_cols, index=one_drift_res.groupby(by=ind_cols, ).max().index)

for ind in one_drift_res_agg.index:
    sub = one_drift_res[(one_drift_res[ind_cols[0]] == ind[0]) & (one_drift_res[ind_cols[1]] == ind[1])].drop_duplicates(
        subset=['lr_test_score'], keep='last'
    )
    if list(sub.stat_test.unique()) != [s for s in stat_tests if s != 'default']:
        print(ind, 'has not all tests: ', list(sub.stat_test.unique()))

    lr_max_ind, pol_max_ind = sub[['lr_test_score', 'pol_lr_test_score']].idxmax()
    if lr_max_ind != pol_max_ind:
        if sub.loc[lr_max_ind, 'lr_test_score'] - sub.loc[pol_max_ind, 'lr_test_score'
        ] >= sub.loc[pol_max_ind, 'pol_lr_test_score'] - sub.loc[lr_max_ind, 'pol_lr_test_score']:
            lr_max_ind = lr_max_ind
        else:
            lr_max_ind = pol_max_ind

    one_drift_res_agg.loc[ind] = sub.loc[lr_max_ind, not_ind_cols]

one_drift_res_agg


# In[520]:


one_drift_res_agg[[c for c in one_drift_res_agg.columns if 'path' not in c.lower()]].style.background_gradient(subset=['lr_train_score', 'lr_test_score', 'pol_lr_train_score', 'pol_lr_test_score'], cmap ='coolwarm', vmin=-1, vmax=1, axis=0)


# In[414]:


one_drift_res[(one_drift_res.dataset_name == 'not-sampled_no-ref-window') & (one_drift_res.data_type_name == 'get_by_feature_score')]


# In[417]:


one_drift_res[(one_drift_res.dataset_name == 'not-sampled_no-ref-window')].drop_duplicates(subset=['lr_test_score'], keep='last')


# In[397]:


one_drift_res[(one_drift_res.dataset_name == 'not-sampled_no-ref-window') & (one_drift_res.data_type_name == 'get_by_feature_score')].idxmax()


# In[394]:


one_drift_res[(one_drift_res.dataset_name == 'not-sampled_no-ref-window') & (one_drift_res.data_type_name == 'get_by_feature_score')]


# ## Getting multi test

# In[317]:


orig_df[stat_tests].apply(lambda x: list(map(get_weighted_thresh_poly, x)))


# In[548]:


def get_models_poly_drift_results(df: pd.DataFrame,
                       rewards_df: pd.DataFrame,
                       features_l: List[str],
                       target: str,
                       stat_ts: List[str],
                       save_path: str, get_func: Callable) -> pd.DataFrame:
    df_4_train = df[['cell_id']].copy()
    df_4_train[stat_ts] = df[stat_ts].apply(lambda x: list(map(get_func, x)))
    f_size = len(features_l)

    results = []
    train_df = pd.DataFrame(index=df_4_train['cell_id'])

    for st in stat_ts:
        temp_df = pd.DataFrame(df_4_train[st].tolist(), index=df_4_train['cell_id'])
        # print(f'[{st}] /t train columns: ', train_df.columns)
        train_df = pd.merge(train_df, temp_df, left_on='cell_id', right_on='cell_id', how='inner', suffixes=('', '_' + st))

    train_df = train_df.merge(rewards_df[['cell_id', target]], left_on='cell_id', right_on='cell_id', how='inner')
    features_l = [f for f in train_df.columns if f not in [target, 'cell_id']]
    print(features_l)
    # print(train_df)
    X_train, X_test, y_train, y_test = train_test_split(train_df[features_l], train_df[target], test_size=0.3, random_state=3407)
    print('X_tr: ', X_train.shape, 'y_te: ', y_test.shape)

    # Linear Regression
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    lr_train_score = model_lr.score(X_train, y_train)
    lr_test_score = model_lr.score(X_test, y_test)
    # dumping model
    lr_path = osp.join(save_path, f'lr_all_stats.joblib')
    dump(model_lr, lr_path)

    # Polynomial
    transformer = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_polynom = transformer.fit_transform(X_train)
    X_test_polynom = transformer.transform(X_test)
    model_pol_lr = LinearRegression().fit(X_train_polynom, y_train)
    pr_train_score = model_pol_lr.score(X_train_polynom, y_train)
    pr_test_score = model_pol_lr.score(X_test_polynom, y_test)
    # dumping model
    pol_lr_path = osp.join(save_path, f'pol_lr_all_stats.joblib')
    trans_path = osp.join(save_path, f'transformer_all_stats.joblib')
    dump(model_pol_lr, pol_lr_path)
    dump(transformer, trans_path)

    # Cross Decomposition
    reg = PLSRegression(n_components=f_size)
    reg.fit(X_train, y_train.to_numpy().ravel())
    plsr_train_score = reg.score(X_train, y_train.to_numpy().ravel())
    plsr_test_score = reg.score(X_test, y_test.to_numpy().ravel())
    # dumping model
    plsr_path = osp.join(save_path, f'plsr_all_stats.joblib')
    dump(reg, plsr_path)

    # save results
    results.append((lr_train_score, lr_test_score,
                    pr_train_score, pr_test_score,
                    plsr_train_score, plsr_test_score,
                    lr_path,
                    pol_lr_path,
                    trans_path,
                    plsr_path))

    res_df = pd.DataFrame(results, columns=['lr_train_score', 'lr_test_score',
                                            'pol_lr_train_score', 'pol_lr_test_score',
                                            'plsr_train_score', 'plsr_test_score',
                                            'lr_path',
                                            'pol_lr_path',
                                            'trans_path',
                                            'plsr_path'])

    return res_df


# In[432]:


path = '/home/rid/Projects/Study/Magister/Dyploma/Bachelor/data/generated/models/poly_drift_0'
get_models_poly_drift_results(orig_df, rewards, features, label, stat_tests, path, get_by_feature_score)


# ### poly drift all run

# In[549]:


datasets_list = [('not-sampled_no-ref-window', 'data/generated/drift/by_cell_agent/run_7/by_train_regressive__sampled-drift-None_no-window_.csv'),
             ('sampled_no-ref-window', 'data/generated/drift/by_cell_agent/run_8/by_train_regressive__sampled-drift-1000_no-window_.csv'),
             ('not-sampled_ref-window-1k', 'data/generated/drift/by_cell_agent/run_9/by_train_regressive_sampled_ref_sampled_drift_None_window_1k_.csv'),
             ('sampled_ref-window-1k', 'data/generated/drift/by_cell_agent/run_10/by_train_regressive_sampled_ref_sampled_drift_1000_window_1k_.csv')]
path = 'data/generated/models/poly_drift_1'
label = 'quality_avg'
features = ['HR Usage Rate',
 'TCH Blocking Rate, BH',
 'Number of Available\nTCH',
 'TCH Traffic (Erl), BH',
 'Param 1',
 'Param 2']
poly_drift_res = pd.DataFrame()

for dataset_name, dataset_path in datasets_list:
    dataset = pd.read_csv(dataset_path, index_col=0)

    for form_data_func in [get_by_feature_score, get_weighted_poly, get_thresh_poly, get_weighted_thresh_poly]:
        data_type_name = form_data_func.__name__

        # regress_df = dataset[['cell_id']].copy()
        # regress_df[stat_tests] = dataset[stat_tests].apply(lambda x: list(map(form_data_func, x)))
        # creating path to save data
        saving_path = osp.join(path, dataset_name, data_type_name)
        Path(saving_path).mkdir(exist_ok=True, parents=True)

        res = get_models_poly_drift_results(dataset.copy(), rewards, features, label, stat_tests, saving_path, form_data_func)
        res['dataset_name'] = [dataset_name] * len(res)
        res['data_type_name'] = [data_type_name] * len(res)

        poly_drift_res = pd.concat([poly_drift_res, res])

poly_drift_res.to_csv(osp.join(path, 'poly_drift_res.csv'))
poly_drift_res


# In[511]:


poly_drift_res.groupby(by=['dataset_name', 'data_type_name']).max()


# In[512]:


ind_cols = ['dataset_name', 'data_type_name']
not_ind_cols = [c for c in poly_drift_res.columns if c not in ind_cols]
poly_drift_res_agg = pd.DataFrame(columns=not_ind_cols, index=poly_drift_res.groupby(by=ind_cols, ).max().index)

for ind in poly_drift_res_agg.index:
    sub = poly_drift_res[(poly_drift_res[ind_cols[0]] == ind[0]) & (poly_drift_res[ind_cols[1]] == ind[1])].drop_duplicates(
        subset=['lr_test_score'], keep='last'
    )

    lr_max_ind, pol_max_ind = sub[['lr_test_score', 'pol_lr_test_score']].idxmax()
    if lr_max_ind != pol_max_ind:
        if sub.loc[lr_max_ind, 'lr_test_score'] - sub.loc[pol_max_ind, 'lr_test_score'
        ] >= sub.loc[pol_max_ind, 'pol_lr_test_score'] - sub.loc[lr_max_ind, 'pol_lr_test_score']:
            lr_max_ind = lr_max_ind
        else:
            lr_max_ind = pol_max_ind

    poly_drift_res_agg.loc[ind] = sub.loc[lr_max_ind, not_ind_cols]

poly_drift_res_agg


# In[531]:


poly_drift_res_agg[[c for c in poly_drift_res_agg.columns if 'path' not in c]].style.background_gradient(subset=['lr_train_score', 'lr_test_score', 'pol_lr_train_score', 'pol_lr_test_score', 'plsr_train_score', 'plsr_test_score'], cmap ='coolwarm', vmin=-1, vmax=1, axis=0)


# In[426]:


one_drift_res[one_drift_res.stat_test == 'wasserstein']


# In[172]:


df_to_train = orig_df[['cell_id']].copy()

df_to_train[stat_tests] = orig_df[stat_tests].apply(lambda x: list(map(get_by_feature_score, x)))
df_to_train


# In[26]:


pd.DataFrame(df_to_train['default'].tolist(), index=df_to_train['cell_id'])


# In[181]:


tmp_res = pd.DataFrame(index=df_to_train['cell_id'])

for st in stat_tests:
    tmp = pd.DataFrame(df_to_train[st].tolist(), index=df_to_train['cell_id'])
    tmp_res = pd.merge(tmp_res, tmp, left_on='cell_id', right_on='cell_id', how='inner', suffixes=('', '_' + st))

tmp_res


# ## Test on linear regression

# In[157]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from joblib import dump, load
import os.path as osp
import numpy as np


# In[100]:


save_path = '/home/rid/Projects/Study/Magister/Dyploma/Bachelor/data/generated/models'


# In[322]:


df_default = pd.DataFrame(df_to_train['default'].tolist(), index=df_to_train['cell_id'])
df_default = df_default.merge(rewards[['cell_id', 'quality_avg']], left_on='cell_id', right_on='cell_id', how='inner')
df_default


# In[323]:


features = [f for f in df_default.columns if f not in ['cell_id', 'quality_avg']]
label = ['quality_avg']


# In[327]:


X_train, X_test, y_train, y_test = train_test_split(df_default[features], df_default[label], test_size=0.3, random_state=3407)


# In[328]:


X_train


# In[329]:


y_test


# ## LinearRegression

# In[330]:


model_lr = LinearRegression()


# In[331]:


model_lr.fit(X_train, y_train)


# In[332]:


model_lr.score(X_train, y_train)


# In[333]:


model_lr.score(X_test, y_test)


# In[334]:


model_lr.coef_


# In[335]:


model_lr.intercept_


# In[101]:


# dump(model_lr, osp.join(save_path, 'model_lr_69.joblib'))


# ## PolynomialFeatures

# In[336]:


transformer = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)


# In[337]:


X_train_polynom = transformer.fit_transform(X_train)
X_test_polynom = transformer.transform(X_test)


# In[338]:


model_pol_lr = LinearRegression().fit(X_train_polynom, y_train)


# In[339]:


model_pol_lr.score(X_train_polynom, y_train)


# In[340]:


model_pol_lr.score(X_test_polynom, y_test)


# In[341]:


model_pol_lr.coef_


# In[342]:


model_lr.intercept_


# In[102]:


# dump(model_pol_lr, osp.join(save_path, 'model_pol_lr_76.joblib'))


# ## PLSRegression

# In[343]:


from sklearn.cross_decomposition import PLSRegression


# In[344]:


reg = PLSRegression(n_components=6)


# In[348]:


reg.fit(X_train_polynom, y_train.to_numpy().ravel())


# In[350]:


reg.score(X_train_polynom, y_train.to_numpy().ravel())


# In[352]:


reg.score(X_test_polynom, y_test.to_numpy().ravel())


# In[547]:


pd.DataFrame(reg.coef_.T).sort_values(by=0).style.background_gradient(cmap ='coolwarm', axis=0)


# In[543]:


np.argpartition(np.abs(list(reg.coef_)), -6)[-6:]


# ## Model's weight visualise

# In[550]:


fs = ['HR Usage Rate', 'Number of Available\nTCH', 'Param 1', 'Param 2', 'TCH Blocking Rate, BH', 'TCH Traffic (Erl), BH', 'HR Usage Rate_ks', 'Number of Available\nTCH_ks', 'Param 1_ks', 'Param 2_ks', 'TCH Blocking Rate, BH_ks', 'TCH Traffic (Erl), BH_ks', 'HR Usage Rate_wasserstein', 'Number of Available\nTCH_wasserstein', 'Param 1_wasserstein', 'Param 2_wasserstein', 'TCH Blocking Rate, BH_wasserstein', 'TCH Traffic (Erl), BH_wasserstein', 'HR Usage Rate_kl_div', 'Number of Available\nTCH_kl_div', 'Param 1_kl_div', 'Param 2_kl_div', 'TCH Blocking Rate, BH_kl_div', 'TCH Traffic (Erl), BH_kl_div', 'HR Usage Rate_psi', 'Number of Available\nTCH_psi', 'Param 1_psi', 'Param 2_psi', 'TCH Blocking Rate, BH_psi', 'TCH Traffic (Erl), BH_psi', 'HR Usage Rate_jensenshannon', 'Number of Available\nTCH_jensenshannon', 'Param 1_jensenshannon', 'Param 2_jensenshannon', 'TCH Blocking Rate, BH_jensenshannon', 'TCH Traffic (Erl), BH_jensenshannon', 'HR Usage Rate_cramer_von_mises', 'Number of Available\nTCH_cramer_von_mises', 'Param 1_cramer_von_mises', 'Param 2_cramer_von_mises', 'TCH Blocking Rate, BH_cramer_von_mises', 'TCH Traffic (Erl), BH_cramer_von_mises', 'HR Usage Rate_hellinger', 'Number of Available\nTCH_hellinger', 'Param 1_hellinger', 'Param 2_hellinger', 'TCH Blocking Rate, BH_hellinger', 'TCH Traffic (Erl), BH_hellinger', 'HR Usage Rate_ed', 'Number of Available\nTCH_ed', 'Param 1_ed', 'Param 2_ed', 'TCH Blocking Rate, BH_ed', 'TCH Traffic (Erl), BH_ed', 'HR Usage Rate_t_test', 'Number of Available\nTCH_t_test', 'Param 1_t_test', 'Param 2_t_test', 'TCH Blocking Rate, BH_t_test', 'TCH Traffic (Erl), BH_t_test']


# In[634]:


def coefs_viz(paths: pd.Series, m_cols: List[str]) -> pd.DataFrame:
    c_df = pd.DataFrame()

    for p in paths:
        m = load(p)
        if len(m.coef_) != 1:
            m.coef_ = m.coef_.reshape(1, -1)

        c_df = pd.concat([c_df, pd.DataFrame(m.coef_, columns=m_cols, index=[p])],)

    return c_df


# ### PLSR

# In[635]:


plsr_res = coefs_viz(poly_drift_res.plsr_path, fs)
plsr_res


# In[569]:


plsr_res.reset_index(drop=True).style.background_gradient(cmap ='coolwarm', axis=1)


# In[608]:


def get_top(df: pd.DataFrame, n = 6):
    top = pd.DataFrame(np.zeros((len(df.columns))), index=df.columns, columns=[f'number_in_top_{n}'])
    # print(top)
    for i in df.index:
        for j in df.loc[i].abs().sort_values(ascending=False)[:n].index:
            top.loc[j] = top.loc[j] +1

    return top.sort_values(by=top.columns[0], ascending=False)


# In[607]:


get_top(plsr_res, 7)


# ### LinearR

# In[636]:


plr_res = coefs_viz(poly_drift_res.lr_path, fs)
plr_res


# In[638]:


get_top(plr_res, 6)


# In[641]:


def get_top_weighted(df: pd.DataFrame, n = 6):
    top = pd.DataFrame(np.zeros((len(df.columns))), index=df.columns, columns=[f'number_in_top_{n}'])
    # print(top)
    for i in df.index:
        print(df.loc[i].abs().sort_values(ascending=False)[:n].index)
        for w, j in enumerate(df.loc[i].abs().sort_values(ascending=False)[:n].index):
            top.loc[j] = top.loc[j] + (n - w)

    return top.sort_values(by=top.columns[0], ascending=False)


# In[642]:


get_top_weighted(plr_res, 6)


# In[643]:


get_top_weighted(plsr_res, 6)


# In[ ]:




