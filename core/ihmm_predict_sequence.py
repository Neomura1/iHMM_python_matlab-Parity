"""Posterior predictive sampling of state sequences for new data."""

from __future__ import annotations

from emissions.gauss_diag_loglik import gauss_diag_loglik
from emissions.poisson_loglik import poisson_loglik
from samplers.ffbs_slice_sample_z import ffbs_slice_sample_z
from core.ihmm_posterior_summary import ihmm_posterior_summary


def ihmm_predict_sequence(Ynew, samples, opt):
    summary = ihmm_posterior_summary(samples, opt)
    Pi = summary['Pi_mean']
    theta = samples['theta'][-1]
    family = opt['emission']['family']
    if family == 'gauss_diag':
        ll_fun = gauss_diag_loglik
    elif family == 'poisson':
        ll_fun = poisson_loglik
    else:
        raise ValueError(f'unknown emission family {family}')

    preds = []
    for y in Ynew:
        loglik = ll_fun(y, theta)
        z = ffbs_slice_sample_z(loglik, Pi)
        preds.append(z)
    return preds

