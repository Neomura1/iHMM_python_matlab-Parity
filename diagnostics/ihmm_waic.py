"""Compute a coarse WAIC from joint log-likelihood samples.

Note: This treats each entire sequence set as one data point, which is a
coarse approximation for HMMs. For finer WAIC, marginal per-timepoint loglik
would be required.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def ihmm_waic(samples: Dict) -> Dict[str, float]:
    """Compute WAIC from joint log-likelihood samples.

    WAIC = -2 * (lppd - p_waic) where
      - lppd = log average likelihood across posterior samples
      - p_waic = variance of pointwise log-likelihood (here approximated by var of joint ll)

    Parameters
    ----------
    samples : dict
        Posterior samples with key 'loglik' (list of floats).

    Returns
    -------
    out : dict
        Dictionary with keys 'waic', 'lppd', 'p_waic', 'S'.
    """

    if not samples or not samples.get('loglik'):
        return {'waic': float('nan'), 'lppd': float('nan'), 'p_waic': float('nan'), 'S': 0}

    ll = np.asarray(samples['loglik'], dtype=float)
    S = ll.shape[0]
    # lppd = log( (1/S) * sum_s exp(ll_s) )
    m = ll.max()
    lppd = float(m + np.log(np.exp(ll - m).mean()))
    p_waic = float(ll.var(ddof=1)) if S > 1 else 0.0
    waic = float(-2.0 * (lppd - p_waic))
    return {'waic': waic, 'lppd': lppd, 'p_waic': p_waic, 'S': int(S)}

