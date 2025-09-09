"""Direct assignment Gibbs sampler for the iHMM."""

from __future__ import annotations

import numpy as np

from emissions.gauss_diag_loglik import gauss_diag_loglik
from emissions.gauss_diag_suffstats import gauss_diag_suffstats
from emissions.gauss_diag_posterior import gauss_diag_posterior
from emissions.gauss_diag_sample import gauss_diag_sample
from emissions.poisson_loglik import poisson_loglik
from emissions.poisson_suffstats import poisson_suffstats
from emissions.poisson_posterior import poisson_posterior
from emissions.poisson_sample import poisson_sample
from hdp.crf_customer_counts import crf_customer_counts
from hdp.crf_table_counts import crf_table_counts
from samplers.sample_beta_sticks import sample_beta_sticks
from samplers.resample_row_transitions import resample_row_transitions
from samplers.ffbs_slice_sample_z import ffbs_slice_sample_z


def sampler_da(Y, state, opt):
    family = opt['emission']['family']
    if family == 'gauss_diag':
        loglik_fun = gauss_diag_loglik
        suffstats_fun = gauss_diag_suffstats
        post_fun = gauss_diag_posterior
        sample_fun = gauss_diag_sample
    elif family == 'poisson':
        loglik_fun = poisson_loglik
        suffstats_fun = poisson_suffstats
        post_fun = poisson_posterior
        sample_fun = poisson_sample
    else:
        raise ValueError(f'unknown emission family {family}')

    # Sample state sequences
    z_list = []
    for y in Y:
        loglik = loglik_fun(y, state['theta'])
        z = ffbs_slice_sample_z(loglik, state['Pi'])
        z_list.append(z)
    state['z'] = z_list

    # Customer and table counts
    counts = crf_customer_counts(z_list)
    state['counts'] = counts
    tables = crf_table_counts(counts, state['alpha'], state.get('kappa', 0.0))
    state['tables'] = tables

    # Global sticks and transitions
    beta = sample_beta_sticks(tables, state['gamma'])
    state['beta'] = beta
    state['Pi'] = resample_row_transitions(counts, state['alpha'], state.get('kappa', 0.0), beta)

    # Emission parameters
    K = len(beta)
    if family == 'gauss_diag':
        D = Y[0].shape[1]
        SS_tot = {'n': np.zeros(K, dtype=int),
                  'y_sum': np.zeros((K, D)),
                  'y2_sum': np.zeros((K, D))}
        for y, z in zip(Y, z_list):
            SS = suffstats_fun(y, z, K)
            SS_tot['n'] += SS['n']
            SS_tot['y_sum'] += SS['y_sum']
            SS_tot['y2_sum'] += SS['y2_sum']
    else:  # poisson
        SS_tot = {'n': np.zeros(K, dtype=int), 'y_sum': np.zeros(K)}
        for y, z in zip(Y, z_list):
            SS = suffstats_fun(y, z, K)
            SS_tot['n'] += SS['n']
            SS_tot['y_sum'] += SS['y_sum']

    posts = post_fun(SS_tot, opt['emission']['prior'])
    state['theta'] = [sample_fun(p) for p in posts]

    return state

