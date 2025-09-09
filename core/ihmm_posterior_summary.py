"""Compute simple posterior summaries from collected samples."""

from __future__ import annotations

import numpy as np


def ihmm_posterior_summary(samples, opt):
    if not samples['beta']:
        return {}
    beta_mean = np.mean(np.vstack(samples['beta']), axis=0)
    Pi_mean = np.mean(np.stack(samples['Pi']), axis=0)

    summary = {'beta_mean': beta_mean, 'Pi_mean': Pi_mean}

    if samples['theta']:
        family = opt['emission']['family']
        if family == 'gauss_diag':
            mu = np.array([[th['mu'] for th in thetas] for thetas in samples['theta']])
            sigma2 = np.array([[th['sigma2'] for th in thetas] for thetas in samples['theta']])
            mu_mean = mu.mean(axis=0).squeeze()
            sigma2_mean = sigma2.mean(axis=0).squeeze()
            order = np.argsort(mu_mean)
            summary['mu_mean'] = mu_mean[order]
            summary['sigma2_mean'] = sigma2_mean[order]
            summary['beta_mean'] = beta_mean[order]
            summary['Pi_mean'] = Pi_mean[order][:, order]
        elif family == 'poisson':
            lam = np.array([[th['lam'] for th in thetas] for thetas in samples['theta']])
            lam_mean = lam.mean(axis=0).squeeze()
            order = np.argsort(lam_mean)
            summary['lambda_mean'] = lam_mean[order]
            summary['beta_mean'] = beta_mean[order]
            summary['Pi_mean'] = Pi_mean[order][:, order]

    return summary

