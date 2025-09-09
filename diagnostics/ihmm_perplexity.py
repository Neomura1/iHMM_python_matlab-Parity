"""Compute a simple training perplexity estimate from saved log-likelihoods."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def ihmm_perplexity(Y: List[np.ndarray], samples: Dict) -> Dict[str, float]:
    """Estimate (training) perplexity from collected log-likelihood samples.

    This uses the joint log-likelihood per sample (already computed in
    ``samples['loglik']``) and normalizes by total number of observations.

    Parameters
    ----------
    Y : list of ndarray
        Observation sequences, each with shape (T_i, D).
    samples : dict
        Posterior samples with key 'loglik' (list of floats).

    Returns
    -------
    out : dict
        Dictionary with keys 'tokens', 'mean_loglik', 'perplexity'.
    """

    if not samples or not samples.get('loglik'):
        return {'tokens': 0, 'mean_loglik': float('nan'), 'perplexity': float('nan')}

    total_tokens = int(sum(y.shape[0] for y in Y))
    loglik = np.asarray(samples['loglik'], dtype=float)
    mean_ll = float(loglik.mean())
    # Training perplexity proxy (sequence-level): exp(-mean_ll / tokens)
    ppl = float(np.exp(-mean_ll / max(total_tokens, 1)))
    return {'tokens': total_tokens, 'mean_loglik': mean_ll, 'perplexity': ppl}

