"""Compute simple posterior summaries from collected samples."""

from __future__ import annotations

import numpy as np


def ihmm_posterior_summary(samples, opt):
    """Compute simple posterior summaries from collected samples.

    This version is robust to varying ``K`` across saved samples (keeps a
    consistent ``K`` based on the last sample) and avoids 0-D shapes when
    ``K=1``/``D=1`` by using per-state scores for ordering.
    """

    if not samples['beta']:
        return {}

    # Choose a reference K (use K from the last saved draw)
    K_ref = len(samples['beta'][-1])
    keep_idx = []
    for i, b in enumerate(samples['beta']):
        if len(b) != K_ref:
            continue
        Pi_i = samples['Pi'][i]
        if Pi_i.shape != (K_ref, K_ref):
            continue
        if samples['theta'] and len(samples['theta'][i]) != K_ref:
            continue
        keep_idx.append(i)

    # If nothing matches (unlikely), fall back to the last draw
    if not keep_idx:
        keep_idx = [len(samples['beta']) - 1]

    beta_arr = np.vstack([samples['beta'][i] for i in keep_idx])
    Pi_arr = np.stack([samples['Pi'][i] for i in keep_idx])
    beta_mean = beta_arr.mean(axis=0)
    Pi_mean = Pi_arr.mean(axis=0)

    summary = {'beta_mean': beta_mean, 'Pi_mean': Pi_mean}

    if samples['theta']:
        family = opt['emission']['family']
        if family == 'gauss_diag':
            # shape: (n_keep, K_ref, D)
            mu = np.array([[th['mu'] for th in samples['theta'][i]] for i in keep_idx])
            sigma2 = np.array([[th['sigma2'] for th in samples['theta'][i]] for i in keep_idx])
            mu_mean = mu.mean(axis=0)       # (K_ref, D) or (K_ref,) or scalar
            sigma2_mean = sigma2.mean(axis=0)

            # Build a 1-D score per state for ordering
            if np.ndim(mu_mean) == 0:  # scalar (K=1, D=1)
                mu_score = np.array([float(mu_mean)])
            elif mu_mean.ndim == 1:    # (K_ref,)
                mu_score = mu_mean
            else:                      # (K_ref, D)
                mu_score = mu_mean.mean(axis=1)

            order = np.argsort(mu_score)

            # Reorder emission means/vars preserving their dimensionality
            if np.ndim(mu_mean) == 0:
                mu_mean_ord = np.atleast_1d(mu_mean)
            elif mu_mean.ndim == 1:
                mu_mean_ord = mu_mean[order]
            else:
                mu_mean_ord = mu_mean[order, :]

            if np.ndim(sigma2_mean) == 0:
                sigma2_mean_ord = np.atleast_1d(sigma2_mean)
            elif sigma2_mean.ndim == 1:
                sigma2_mean_ord = sigma2_mean[order]
            else:
                sigma2_mean_ord = sigma2_mean[order, :]

            summary['mu_mean'] = mu_mean_ord
            summary['sigma2_mean'] = sigma2_mean_ord
            summary['beta_mean'] = beta_mean[order]
            summary['Pi_mean'] = Pi_mean[order][:, order]
        elif family == 'poisson':
            lam = np.array([[th['lam'] for th in samples['theta'][i]] for i in keep_idx])  # (n_keep, K_ref)
            lam_mean = lam.mean(axis=0)  # (K_ref,)
            order = np.argsort(lam_mean)
            summary['lambda_mean'] = lam_mean[order]
            summary['beta_mean'] = beta_mean[order]
            summary['Pi_mean'] = Pi_mean[order][:, order]

    # Prune near-zero weight states from the summary for clearer reporting
    # Default threshold can be overridden via opt['prune_beta_thresh']
    thresh = float(opt.get('prune_beta_thresh', 1e-3))
    beta_mean = summary['beta_mean']
    Pi_mean = summary['Pi_mean']
    active = (beta_mean > thresh)
    if active.ndim == 0:
        active = np.array([bool(active)])
    if not np.any(active):
        # Keep at least the heaviest state if all fall below threshold
        keep_idx = [int(np.argmax(beta_mean))]
        active = np.zeros(beta_mean.shape, dtype=bool)
        active[keep_idx] = True

    # Apply mask consistently
    summary['beta_mean'] = beta_mean[active]
    summary['Pi_mean'] = Pi_mean[active][:, active]
    if 'mu_mean' in summary:
        mu_mean = summary['mu_mean']
        if np.ndim(mu_mean) == 0:
            summary['mu_mean'] = np.atleast_1d(mu_mean)
        elif mu_mean.ndim == 1:
            summary['mu_mean'] = mu_mean[active]
        else:
            summary['mu_mean'] = mu_mean[active, :]
    if 'sigma2_mean' in summary:
        s2 = summary['sigma2_mean']
        if np.ndim(s2) == 0:
            summary['sigma2_mean'] = np.atleast_1d(s2)
        elif s2.ndim == 1:
            summary['sigma2_mean'] = s2[active]
        else:
            summary['sigma2_mean'] = s2[active, :]
    if 'lambda_mean' in summary:
        lam = summary['lambda_mean']
        summary['lambda_mean'] = lam[active]

    summary['K_effective'] = int(np.sum(active))
    summary['active_mask'] = active

    return summary
