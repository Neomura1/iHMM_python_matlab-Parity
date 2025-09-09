import os
import sys
import numpy as np

# allow running from the repository root without installing as a package
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.ihmm_fit import ihmm_fit


def ex_poisson_counts(seed: int = 0):
    """Run a small iHMM on Poisson count data using the direct-assignment sampler."""

    rng = np.random.default_rng(seed)
    T = 150
    lams = np.array([2.0, 8.0])
    z_true = rng.integers(0, len(lams), size=T)
    y = rng.poisson(lams[z_true])
    Y = [y[:, None]]

    opt = {
        'seed': seed,
        'K_init': 4,
        'alpha': 6.0,
        'gamma': 6.0,
        'n_iter': 30,
        'burnin': 15,
        'thin': 1,
        'emission': {
            'family': 'poisson',
            'prior': {
                'a0': 1.0,
                'b0': 1.0,
            },
        },
    }

    state, samples, summary = ihmm_fit(Y, opt)
    print('beta_mean:', summary['beta_mean'])
    print('Pi_mean:', summary['Pi_mean'])
    print('last loglik:', samples['loglik'][-1])


if __name__ == '__main__':
    ex_poisson_counts()
