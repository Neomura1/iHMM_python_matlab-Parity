"""Run Gibbs sampling iterations and collect posterior samples."""

from __future__ import annotations

from core.ihmm_init_model import ihmm_init_model
from samplers.sampler_da import sampler_da
from samplers.calc_loglik_current import calc_loglik_current


def ihmm_mainloop(Y, state, opt):
    n_iter = opt.get('n_iter', 100)
    burnin = opt.get('burnin', 0)
    thin = opt.get('thin', 1)

    samples = {'beta': [], 'Pi': [], 'theta': [], 'z': [], 'loglik': []}
    for it in range(n_iter):
        state = sampler_da(Y, state, opt)
        if it >= burnin and ((it - burnin) % thin == 0):
            samples['beta'].append(state['beta'].copy())
            samples['Pi'].append(state['Pi'].copy())
            samples['theta'].append([dict(t) for t in state['theta']])
            samples['z'].append([z.copy() for z in state['z']])
            samples['loglik'].append(calc_loglik_current(Y, state, opt))
    return state, samples

