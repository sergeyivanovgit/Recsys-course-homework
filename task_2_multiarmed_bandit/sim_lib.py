
import numpy as np
import pandas as pd

from scipy.stats import beta, uniform, bernoulli, expon
from typing import Callable

ALPHA = 1
BETA = 20
MU = 10 ** 4
MONITORING_FREQ = 10 ** 4


def generate_new_banner(n, a=ALPHA, b=BETA, mu=MU):
    p = beta.rvs(a, b, size=n)
    lifetimes = expon.rvs(scale=mu, size=n)
    return p, lifetimes


def simulation(policy: Callable, n=10 ** 6, initial_banners=9):
    state = pd.DataFrame(np.zeros((initial_banners, 4)), columns=['impressions', 'clicks', 'lifetime', 'p'])
    state['p'], state['lifetime'] = generate_new_banner(initial_banners)
    regret = 0
    max_index = initial_banners
    borning_rate = initial_banners*(1-np.exp(-1/MU))

    for i in range(n):
        if uniform.rvs() < borning_rate or state.shape[0] < 2:
            p, lifetime = generate_new_banner(1)
            new_banner = pd.DataFrame({'impressions': 0, 'clicks': 0, 'lifetime': lifetime, 'p': p}, index=[max_index])
            state = pd.concat([state, new_banner], verify_integrity=True)
            max_index += 1

        index = policy(state[['impressions', 'clicks']].copy())

        assert index in state.index, 'Error, wrong action number'

        p = state.loc[index, 'p']
        reward = bernoulli.rvs(p)
        state.loc[index, 'impressions'] += 1
        state.loc[index, 'clicks'] += reward
        regret = regret + max(state['p']) - p

        state['lifetime'] = state['lifetime'] - 1
        state = state[state['lifetime'] > 0]

        if not i % MONITORING_FREQ:
            print('{} impressions have been simulated'.format(i + 1))

    return {'regret': regret, 'rounds': n, 'total_banners': max_index, 'history': state}