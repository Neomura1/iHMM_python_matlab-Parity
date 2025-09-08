"""Compute simple posterior summaries from collected samples."""

from __future__ import annotations

import numpy as np


def ihmm_posterior_summary(samples, opt):
    if not samples['beta']:
        return {}
    beta_mean = np.mean(np.vstack(samples['beta']), axis=0)
    Pi_mean = np.mean(np.stack(samples['Pi']), axis=0)
    summary = {'beta_mean': beta_mean, 'Pi_mean': Pi_mean}
    return summary

