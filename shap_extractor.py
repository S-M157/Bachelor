import pickle

import shap

import glob
import json
import os
from pathlib import Path
from typing import List, Type, Dict, Any, Callable, Tuple

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


def extract_shap(
        data: pd.DataFrame,
        f: Callable[..., np.ndarray],
        results_path: str,
        name: str,
        use_force: bool = False,
        save_shap: bool = True,
        save_observations: bool = False) -> Tuple[str, float]:

    # def f(x):
    #     return agent.act(torch.as_tensor(x, dtype=torch.float32))

    cols = data.columns

    # create results folders
    plots_path = os.path.join(results_path, 'plots')
    shap_path = results_path

    Path(plots_path).mkdir(parents=True, exist_ok=True)
    Path(shap_path).mkdir(parents=True, exist_ok=True)

    # init values
    shap_values = None
    not_done = True
    # setting patience to SHAP calculation error due to values forming in the library
    patience = 1
    fails = 0
    expected_val = None

    # SHAP params
    observs = np.array(data)
    sampled_obs = shap.sample(observs, 1, random_state=42)
    explainer = shap.KernelExplainer(f, sampled_obs, )

    while not_done:
        # SHAP can throw error with matrix sizes (just bag)
        try:
            shap_values = explainer.shap_values(observs, nsamples=100)
            not_done = False
        except ValueError:
            fails += 1
            not_done = fails <= patience

    if shap_values is not None:
        # SHAP distribution
        plt.clf()

        shap.summary_plot(shap_values, observs, feature_names=cols, show=False)

        plt.title(f'{name}')

        plt.savefig(os.path.join(plots_path, f'{name}.png'), bbox_inches='tight')

        # SHAP Force
        if use_force:
            out_plot = shap.plots.force(explainer.expected_value, shap_values, observs, feature_names=cols,
                                        show=False)
            shap.save_html(os.path.join(plots_path, f'{name}.html'), out_plot, full_html=True)

        # store expected values
        expected_val = explainer.expected_value

        # Save SHAP
        if save_shap:
            _save_path = os.path.join(shap_path, f'{name}_shap.pkl')

            try:
                shape_df = pd.DataFrame(shap_values, columns=cols)
                shape_df.to_pickle(_save_path)

            except ValueError as e:

                with open(_save_path, 'wb') as f:
                    pickle.dump(shap_values, f)

        if save_observations:
            obs_df = pd.DataFrame(observs, columns=cols)
            obs_df.to_pickle(shap_path + f'/{name}_obs.pickle')

    return results_path, expected_val

