"""Initialise state dictionary for the iHMM."""

from __future__ import annotations

import numpy as np

from utils.rng_manager import rng_manager
from hdp.stick_breaking import stick_breaking
from hdp.crf_customer_counts import crf_customer_counts
from hdp.crf_table_counts import crf_table_counts
from samplers.resample_row_transitions import resample_row_transitions
from emissions.gauss_diag_init import gauss_diag_init
from emissions.poisson_init import poisson_init


def ihmm_init_model(Y, opt):
    rng_manager(opt.get('seed', 0))
    K = opt.get('K_init', 3)
    alpha = opt.get('alpha', 1.0)
    gamma = opt.get('gamma', 1.0)
    kappa = opt.get('kappa', 0.0)

    beta = stick_breaking(gamma, K)
    Pi = resample_row_transitions(np.zeros((K, K)), alpha, kappa, beta)

    family = opt['emission']['family']
    prior = opt['emission']['prior']
    theta = []
    for _ in range(K):
        if family == 'gauss_diag':
            theta.append(gauss_diag_init(Y[0], prior))
        elif family == 'poisson':
            theta.append(poisson_init(Y[0], prior))
        else:
            raise ValueError(f'unknown emission family {family}')

    z_list = []
    for y in Y:
        T = y.shape[0]
        z_list.append(np.random.choice(K, size=T))

    counts = crf_customer_counts(z_list)
    tables = crf_table_counts(counts, alpha, kappa)

    state = {
        'alpha': alpha,
        'gamma': gamma,
        'kappa': kappa,
        'beta': beta,
        'Pi': Pi,
        'theta': theta,
        'z': z_list,
        'counts': counts,
        'tables': tables,
    }
    return state

