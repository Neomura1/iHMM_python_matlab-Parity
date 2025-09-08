"""High level training entry point for the iHMM."""

from __future__ import annotations

from core.ihmm_init_model import ihmm_init_model
from core.ihmm_mainloop import ihmm_mainloop
from core.ihmm_posterior_summary import ihmm_posterior_summary


def ihmm_fit(Y, opt):
    """Fit an iHMM to observation sequences ``Y``.

    Parameters
    ----------
    Y : list of ndarray
        Observation sequences, each of shape ``(T, D)``.
    opt : dict
        Options and hyperparameters.
    """

    state = ihmm_init_model(Y, opt)
    state, samples = ihmm_mainloop(Y, state, opt)
    summary = ihmm_posterior_summary(samples, opt)
    return state, samples, summary

