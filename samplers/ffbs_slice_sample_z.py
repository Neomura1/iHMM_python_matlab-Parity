"""Forward-filtering backward-sampling for hidden state sequences.

The slice variables are ignored in this simplified implementation but the
signature is kept for parity with the MATLAB version.
"""

from __future__ import annotations

import numpy as np

from utils.logsumexp import logsumexp
from utils.sample_discrete import sample_discrete


def ffbs_slice_sample_z(loglik, trans, slice_u=None, active_set=None):
    loglik = np.asarray(loglik)
    trans = np.asarray(trans)
    K, T = loglik.shape
    log_trans = np.log(np.maximum(trans, 1e-300))

    log_alpha = np.zeros((K, T))
    log_alpha[:, 0] = loglik[:, 0]
    for t in range(1, T):
        temp = log_alpha[:, t - 1][:, None] + log_trans
        log_alpha[:, t] = loglik[:, t] + logsumexp(temp, axis=0)

    z = np.zeros(T, dtype=int)
    log_p = log_alpha[:, -1]
    z[-1] = sample_discrete(np.exp(log_p - logsumexp(log_p)))
    for t in range(T - 2, -1, -1):
        log_p = log_alpha[:, t] + log_trans[:, z[t + 1]]
        z[t] = sample_discrete(np.exp(log_p - logsumexp(log_p)))
    return z

