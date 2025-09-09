import os
import sys
import numpy as np

# allow running from the repository root without installing as a package
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.ihmm_fit import ihmm_fit
from diagnostics.traceplot_scalar import traceplot_scalar
from diagnostics.plot_transition_heatmap import plot_transition_heatmap
from diagnostics.plot_state_sequence import plot_state_sequence
from diagnostics.ihmm_perplexity import ihmm_perplexity
from diagnostics.ihmm_waic import ihmm_waic


def ex_gauss1d_da(seed: int = 0):
    """Run a small iHMM on 1D Gaussian data using the direct-assignment sampler.

    Generates synthetic observations from three Gaussian states, trains the
    iHMM and prints a few posterior summaries.
    """

    rng = np.random.default_rng(seed)
    T = 5000
    mus = np.array([-2.0, 0.0, 2.0])
    sigmas = np.array([0.4, 0.5, 0.4])
    z_true = rng.integers(0, len(mus), size=T)
    y = mus[z_true] + rng.normal(scale=sigmas[z_true])
    Y = [y[:, None]]  # single sequence with shape (T, 1)

    opt = {
        'seed': seed,
        'K_init': 8,
        'alpha': 6.0,
        'gamma': 6.0,
        'kappa': 10.0, 
        'prune_min_count': 0.1*len(Y),
        'verbose': True,
        'print_every': 10,
        'n_iter': 500,
        'burnin': 50,
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
    mask = summary['beta_mean'] > 0.05
    print('mu_mean:', summary.get('mu_mean')[mask])
    print('sigma2_mean:', summary.get('sigma2_mean')[mask])
    print('z sample (first 20):', samples['z'][-1][0][:20])
    print('z true (first 20):', z_true[0:20])

    # Diagnostics: perplexity and WAIC (coarse)
    ppl = ihmm_perplexity(Y, samples)
    waic = ihmm_waic(samples)
    print('perplexity (train):', ppl['perplexity'])
    print('WAIC:', waic['waic'])

    # Diagnostics plots (if matplotlib is available)
    try:
        import matplotlib.pyplot as plt
        # 1) Log-likelihood trace
        traceplot_scalar(samples['loglik'], title='Log-likelihood trace', ylab='log p(Y|θ)')
        # 2) Transition matrix heatmap (posterior mean)
        plot_transition_heatmap(summary['Pi_mean'], title='Mean transition matrix')
        # 3) State-colored observations for the last saved sample
        plot_state_sequence(Y[0], samples['z'][-1][0], title='State-colored observations (last sample)')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print('Plotting skipped:', e)
    


if __name__ == '__main__':
    ex_gauss1d_da()
