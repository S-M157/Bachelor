import torch
from torch.optim import Adam
import numpy as np
import itertools
from copy import deepcopy
from glob import glob

from rl.sac import ReplayBuffer
from helpers import load_agent
from preprocess import preprocess_stats

seed = 10
torch.manual_seed(seed)
np.random.seed(seed)

ac = load_agent('sac_last_60_50d_exp-r.pt', 'pt')
ac_targ = deepcopy(ac)
obs_dim = 6
act_dim = 2
batch_size = 100
update_every = 50
lr = 1e-3
gamma = 0.99
alpha = 0.2
polyak = 0.995

# List of parameters for both Q-networks (save this for convenience)
q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
# Set up optimizers for policy and q-function
pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
q_optimizer = Adam(q_params, lr=lr)
# Experience buffer
replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=int(1e6))


def compute_loss_q(data):
    o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

    q1 = ac.q1(o, a)
    q2 = ac.q2(o, a)

    # Bellman backup for Q functions
    with torch.no_grad():
        # Target actions come from *current* policy
        a2, logp_a2 = ac.pi(o2)

        # Target Q-values
        q1_pi_targ = ac_targ.q1(o2, a2)
        q2_pi_targ = ac_targ.q2(o2, a2)
        q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
        backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

    # MSE loss against Bellman backup
    loss_q1 = ((q1 - backup) ** 2).mean()
    loss_q2 = ((q2 - backup) ** 2).mean()
    loss_q = loss_q1 + loss_q2

    # Useful info for logging
    q_info = dict(Q1Vals=q1.detach().numpy(),
                  Q2Vals=q2.detach().numpy())

    return loss_q, q_info


# Set up function for computing SAC pi loss
def compute_loss_pi(data):
    o = data['obs']
    pi, logp_pi = ac.pi(o)
    q1_pi = ac.q1(o, pi)
    q2_pi = ac.q2(o, pi)
    q_pi = torch.min(q1_pi, q2_pi)

    # Entropy-regularized policy loss
    loss_pi = (alpha * logp_pi - q_pi).mean()

    # Useful info for logging
    pi_info = dict(LogPi=logp_pi.detach().numpy())

    return loss_pi, pi_info


def update(data):
    # First run one gradient descent step for Q1 and Q2
    q_optimizer.zero_grad()
    loss_q, q_info = compute_loss_q(data)
    loss_q.backward()
    q_optimizer.step()

    # Freeze Q-networks so you don't waste computational effort
    # computing gradients for them during the policy learning step.
    for p in q_params:
        p.requires_grad = False

    # Next run one gradient descent step for pi.
    pi_optimizer.zero_grad()
    loss_pi, pi_info = compute_loss_pi(data)
    loss_pi.backward()
    pi_optimizer.step()

    # Unfreeze Q-networks so you can optimize it at next DDPG step.
    for p in q_params:
        p.requires_grad = True

    # Finally, update target networks by polyak averaging.
    with torch.no_grad():
        for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
            # NB: We use an in-place operations "mul_", "add_" to update target
            # params, as opposed to "mul" and "add", which would make new tensors.
            p_targ.data.mul_(polyak)
            p_targ.data.add_((1 - polyak) * p.data)

def reward():
    pass


def train(folder: str):
    columns = ['HR Usage Rate', 'TCH Blocking Rate, BH', 'Number of Available\nTCH',
               'TCH Traffic (Erl), BH', 'Param 1', 'Param 2']
    files = glob(folder+'*.xlsm')
    for i in range(len(files)):
        df = preprocess_stats(files[i], columns)
        df2 = preprocess_stats(files[i+1], columns)
        action = [df2['Param 1']-df['Param 1'], df2['Param 2']-df['Param 2']]
        for j, j2 in zip(df.values, df2.values):
            replay_buffer.store(j, action, reward, j2, done)

    for j in range(update_every):
        batch = replay_buffer.sample_batch(batch_size)
        update(data=batch)
