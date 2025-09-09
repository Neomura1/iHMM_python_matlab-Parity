import os
import sys
import numpy as np

# allow running from the repository root without installing as a package
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.ihmm_fit import ihmm_fit


def ex_gauss1d_da(seed: int = 0):
    """Run a small iHMM on 1D Gaussian data using the direct-assignment sampler.

    Generates synthetic observations from three Gaussian states, trains the
    iHMM and prints a few posterior summaries.
    """

    rng = np.random.default_rng(seed)
    T = 200
    mus = np.array([-2.0, 0.0, 2.0])
    sigmas = np.array([0.3, 0.2, 0.3])
    z_true = rng.integers(0, len(mus), size=T)
    y = mus[z_true] + rng.normal(scale=sigmas[z_true])
    Y = [y[:, None]]  # single sequence with shape (T, 1)

    opt = {
        'seed': seed,
        'K_init': 5,
        'alpha': 6.0,
        'gamma': 6.0,
        'n_iter': 30,
        'burnin': 15,
        'thin': 1,
        'emission': {
            'family': 'gauss_diag',
            'prior': {
                'mu0': np.zeros(1),
                'kappa0': np.ones(1),
                'a0': np.ones(1),
                'b0': np.ones(1),
            },
        },
    }

    state, samples, summary = ihmm_fit(Y, opt)
    print('beta_mean:', summary['beta_mean'])
    print('Pi_mean:', summary['Pi_mean'])
    print('z sample (first 20):', samples['z'][-1][0][:20])


if __name__ == '__main__':
    ex_gauss1d_da()
