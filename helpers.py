import pickle
import numpy as np
import torch


def predict(o, agent):
    return agent.act(torch.as_tensor(o, dtype=torch.float32))


def quality(blocking, ch, traffic, param1, param2, prparam1, prparam2):
    res = 0
    if traffic == 0:
        traffic = 1
    # якщо блокування рівне нулю, параметри зменшились і каналів вдвіче більше, ніж трафіку
    if blocking == 0 and (param1 >= prparam1 and param2 >= prparam2) and ch / traffic >= 2:
        res = 1
    # якщо блокування існує, деякі параметри зменшились і каналів вдвіче більше, ніж трафіку
    elif blocking != 0 and (param1 < prparam1 or param2 < prparam2) and ch / traffic >= 2:
        res = 1
    # якщо блокування рівне нулю, деякі параметри збільшились і каналів не вдвіче більше, ніж трафіку
    elif blocking == 0 and (param1 >= prparam1 or param2 >= prparam2) and ch / traffic <= 2:
        res = 1
    # якщо блокування існує і параметри зменшились
    elif blocking != 0 and (param1 < prparam1 and param2 < prparam2):
        res = 1

    return res


def bad_decision(blocking, channels, traffic, param1, param2, prparam1, prparam2):
    res = 0
    if blocking != 0 and (param1 > prparam1 and param2 > prparam2) and channels / traffic <= 2:
        res = 1
    elif blocking == 0 and (param1 < prparam1 and param2 < prparam2) and channels / traffic >= 2:
        res = 1
    elif prparam1 >= prparam2:
        res = 1
    return res


def load_agent(name, ext):
    if ext == 'pt':
        agent = torch.load(f'agent/{name}')
        agent.eval()
    else:
        with open(f'agent/{name}', 'rb') as f:
            agent = pickle.load(f)
    return agent


def reward_func(hr, blocking):
    return (1 - (0.2 * hr / 100 + 0.8 * np.log(blocking / 100 + 1)) / (0.2 + 0.8 * np.log(2))) * 100


def clip(value, lower=0, upper=100):
    return lower if value < lower else upper if value > upper else value