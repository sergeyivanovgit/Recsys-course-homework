import pandas as pd
import numpy as np
from scipy.stats import beta, uniform, bernoulli, randint, expon

ALPHA = 1
BETA = 20
MU = 10 ** 4
MONITORING_FREQ = 10 ** 4


def generate_new_banner(n, a=ALPHA, b=BETA, mu=MU):
    p = beta.rvs(a, b, size=n)
    lt = expon.rvs(scale=mu, size=n)
    return p, lt


def simulation(policy, n=10 ** 6, initial_banners=10):
    state = pd.DataFrame(np.zeros((initial_banners, 4)), columns=['impressions', 'clicks', 'lifetime', 'p'])
    state['p'], state['lifetime'] = generate_new_banner(initial_banners)
    regret = 0
    max_index = initial_banners
    borning_rate = 7.2 / MU

    for i in range(n):
        if uniform.rvs() < borning_rate or state.shape[0] < 2:
            p, lt = generate_new_banner(1)
            new_banner = pd.DataFrame({'impressions': 0, 'clicks': 0, 'lifetime': lt, 'p': p}, index=[max_index])
            state = pd.concat([state, new_banner], verify_integrity=True)
            max_index += 1

        index = policy(state[['impressions', 'clicks']].copy())

        # assert isinstance(index, int), 'Error, wrong action type'
        assert index in state.index, 'Error, wrong action number'

        p = state.loc[index, 'p']
        reward = bernoulli.rvs(p)
        state.loc[index, 'impressions'] += 1
        state.loc[index, 'clicks'] += reward
        regret = regret + max(state.p) - p

        state['lifetime'] = state['lifetime'] - 1
        state = state.query(""" lifetime>0 """)

        if i % MONITORING_FREQ == 0:
            print('{} impressions have been simulated'.format(i + 1))

    return {'regret': regret, 'rounds': n, 'total_banners': max_index, 'history': state}