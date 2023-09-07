import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
from random import randint


def clip(value, lower=0, upper=100):
    return lower if value < lower else upper if value > upper else value


class SimulatedCustomEnv(Env):

    def __init__(self, env, action_range, series, covariates, n_past, verbose: bool = False):
        # we create an observation space with predefined range
        self.observation_space = env
        self.n_past = n_past
        self._reward = 0
        self.series = series
        self.covariates = covariates
        self.period = 7
        self.done = False
        self.verbose = verbose
        # 'HR Usage Rate', 'TCH Blocking Rate, BH'
        self.current_state = series.head(n_past)
        # 'Number of Available\nTCH', 'TCH Traffic (Erl), BH', 'Param 1',  'Param 2'
        self.cov = covariates.head(n_past)

        # similar to observation, we define action space
        self.action_space = Box(low=-1, high=action_range)
        self.cell_train_index = 0
        self.cell_series_train_index = 0

    def state_eval(self, after, before):
        if not self.done:
            if after[1] > before[1]:
                self.done = True
                self._reward -= 10
            elif self.period == 0:
                self.done = True
                self._reward += 10
            elif after[0] < before[0]:
                self._reward += 5
        else:
            self._reward -= 10

    def set_current_state(self, new_state, state_before, cov_b):
        state = new_state.last_values()
        state[0] = clip(state[0])
        state[1] = clip(state[1])
        cov = self.cov.last_values()
        if all(cov[-2:] < cov_b[-2:]) and state_before[1] < state[1]: state[1] = 0
        self.current_state = self.current_state.append_values(state.reshape(1, len(state))).tail(self.n_past)

    def do_action(self, actions):
        cov = self.cov.last_values()
        # import pdb; pdb.set_trace()
        action_estimate = any((cov[:2] + actions) > 100) or any((cov[:2] + actions) < 0)
        cov[2] = int(clip(cov[2] + actions[0]))
        cov[3] = int(clip(cov[3] + actions[1]))
        if action_estimate: self._reward -= 50
        state = self.current_state.last_values()
        self.current_state = self.current_state.append_values(state.reshape(1, len(state))).tail(self.n_past)
        self.cov = self.cov.append_values(cov.reshape(1, len(cov))).tail(self.n_past)
        return cov[2] >= cov[3]

    def step(self, actions):
        state_before = self.current_state.last_values()
        cov_before = self.cov.last_values()
        if self.verbose:
            print("State U/B before:", state_before)
            print("State before:", cov_before)
            print("Action:", actions * 30)
        #
        self.done = self.do_action(actions * 30)
        if self.verbose:
            print("State after:", self.cov.last_values())

        if self.cov.end_time() != self.current_state.end_time():
            cov = self.cov.last_values()
            # import pdb; pdb.set_trace()
            cov[1] = self.cov.pd_dataframe()['TCH Traffic (Erl), BH'].mean()
            self.cov = self.cov.append_values(cov.reshape(1, len(cov))).tail(self.n_past)

        self.set_current_state(self.observation_space.predict(n=1, series=self.current_state, past_covariates=self.cov,
                                                              verbose=self.verbose), state_before, cov_before)

        if self.verbose:
            print("Next U/B:", self.current_state.last_values())

        self.state_eval(self.current_state.last_values()[:2], state_before)
        self.period -= 1

        return np.concatenate([self.current_state.last_values(), self.cov.last_values()]), self._reward, self.done, {}

    def reset(self):
        self.cell_series_train_index += self.n_past - 1
        if self.cell_series_train_index + self.n_past >= len(self.series):
            self.cell_series_train_index = 0
        index = self.series.time_index
        self._reward = 0
        self.period = 7
        self.done = False
        self.current_state = self.series.slice(index[self.cell_series_train_index],
                                               index[self.cell_series_train_index + self.n_past - 1])
        self.cov = self.covariates.slice(index[self.cell_series_train_index],
                                         index[self.cell_series_train_index + self.n_past - 1])
        # self.quality_before = self.quality_function(self.current_state.last_values()[0], self.current_state.last_values()[1])
        return np.concatenate([self.current_state.last_values(), self.cov.last_values()])


class SimulatedQualEnv(Env):

    def __init__(self, quality_function, env, action_range, series, covariates, n_past, scaler):
        # we create an observation space with predefined range
        self.observation_space = env
        self.n_past = n_past
        self._reward = 0
        self.series = series
        self.covariates = covariates
        self.period = 7
        # 'HR Usage Rate', 'TCH Blocking Rate, BH', 'TCH Traffic (Erl), BH'
        self.current_state = series[0].head(n_past)
        # 'Number of Available\nTCH', 'Param 1',  'Param 2'
        self.cov = covariates[0].head(n_past)
        self.scaler = scaler
        self.quality_function = quality_function
        # similar to observation, we define action space
        self.action_space = Box(low=-1, high=action_range)
        self.done = False
        self.cell_train_index = 0
        self.cell_series_train_index = 0
        self.quality_after = 0

    def set_current_state(self, new_state, state_before, cov_b):
        state = new_state.last_values()
        state[0] = clip(state[0])
        state[1] = clip(state[1])
        cov = self.cov.last_values()
        if all(cov[-2:] < cov_b[-2:]) and state_before[1] < state[1]: state[1] = 0
        self.current_state = self.current_state.append_values(state.reshape(1, len(state))).tail(self.n_past)

    def do_action(self, actions):
        cov = self.cov.last_values()
        action_estimate = any((cov[:2] + actions) > 100) or any((cov[:2] + actions) < 0)
        cov[2] = int(clip(cov[2] + actions[0]))
        cov[3] = int(clip(cov[3] + actions[1]))
        if action_estimate: self._reward -= 30
        state = self.current_state.last_values()
        self.current_state = self.current_state.append_values(state.reshape(1, len(state))).tail(self.n_past)
        self.cov = self.cov.append_values(cov.reshape(1, len(cov))).tail(self.n_past)
        return cov[2] >= cov[3]

    # TODO delete the first day when adding the new
    def step(self, actions):
        state_before = self.current_state.last_values()
        unormalized_state_before = self.scaler.inverse_transform(self.current_state)
        quality_before = self.quality_function(unormalized_state_before.last_values()[0],
                                               unormalized_state_before.last_values()[1])
        cov_before = self.cov.last_values()
        self.done = self.do_action(actions * 30)
        if self.cov.end_time() != self.current_state.end_time():
            cov = self.cov.last_values()
            cov[1] = self.cov.pd_dataframe()['TCH Traffic (Erl), BH'].mean()
            self.cov = self.cov.append_values(cov.reshape(1, len(cov))).tail(self.n_past)

        self.set_current_state(
            self.observation_space.predict(n=1, series=self.current_state, past_covariates=self.cov, verbose=False),
            state_before, cov_before)
        unormalized_state_after = self.scaler.inverse_transform(self.current_state)
        self.quality_after = self.quality_function(unormalized_state_after.last_values()[0],
                                                   unormalized_state_after.last_values()[1])
        self._reward += self.quality_after - quality_before if self.quality_after < 1 else 10
        self.period -= 1
        return np.concatenate([self.current_state.last_values(), self.cov.last_values()]), self._reward, self.done, {}

    def reset(self):
        self.cell_series_train_index += 1
        index = self.series.time_index
        self._reward = 0
        self.period = 7
        self.done = False
        self.current_state = self.series.slice(index[self.cell_series_train_index],
                                               index[self.cell_series_train_index + self.n_past - 1])
        self.cov = self.covariates.slice(index[self.cell_series_train_index],
                                         index[self.cell_series_train_index + self.n_past - 1])
        return np.concatenate([self.current_state.last_values(), self.cov.last_values()])
