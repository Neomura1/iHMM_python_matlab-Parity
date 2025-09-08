"""Compute joint log likelihood of data under current state."""

from __future__ import annotations

import numpy as np

from utils.logsumexp import logsumexp
from emissions.gauss_diag_loglik import gauss_diag_loglik
from emissions.poisson_loglik import poisson_loglik


def calc_loglik_current(Y, state, opt):
    family = opt['emission']['family']
    if family == 'gauss_diag':
        ll_fun = gauss_diag_loglik
    elif family == 'poisson':
        ll_fun = poisson_loglik
    else:
        raise ValueError(f'unknown emission family {family}')

    Pi = state['Pi']
    log_trans = np.log(np.maximum(Pi, 1e-300))
    total = 0.0
    for y in Y:
        loglik = ll_fun(y, state['theta'])
        K, T = loglik.shape
        log_alpha = loglik[:, 0]
        for t in range(1, T):
            temp = log_alpha[:, None] + log_trans
            log_alpha = loglik[:, t] + logsumexp(temp, axis=0)
        total += logsumexp(log_alpha)
    return float(total)

