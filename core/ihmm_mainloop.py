"""Run Gibbs sampling iterations and collect posterior samples."""

from __future__ import annotations

from core.ihmm_init_model import ihmm_init_model
from samplers.sampler_da import sampler_da
from samplers.calc_loglik_current import calc_loglik_current


def ihmm_mainloop(Y, state, opt):
    n_iter = opt.get('n_iter', 100)
    burnin = opt.get('burnin', 0)
    thin = opt.get('thin', 1)
    verbose = bool(opt.get('verbose', False))
    print_every = int(opt.get('print_every', max(1, n_iter // 10)))
    print_loglik = bool(opt.get('print_loglik', True))

    samples = {'beta': [], 'Pi': [], 'theta': [], 'z': [], 'loglik': [], 'K': [], 'prune_info': []}
    if verbose:
        try:
            print(f"[iHMM] start training: n_iter={n_iter}, burnin={burnin}, thin={thin}")
        except Exception:
            pass
    for it in range(n_iter):
        state, diag = sampler_da(Y, state, opt)

        # Determine whether to compute the current log-likelihood
        need_save = it >= burnin and ((it - burnin) % thin == 0)
        need_print = verbose and (it == 0 or it == n_iter - 1 or ((it + 1) % print_every == 0))
        ll_cur = None
        if (need_save or (need_print and print_loglik)):
            ll_cur = calc_loglik_current(Y, state, opt)

        if need_save:
            samples['beta'].append(state['beta'].copy())
            samples['Pi'].append(state['Pi'].copy())
            samples['theta'].append([dict(t) for t in state['theta']])
            samples['z'].append([z.copy() for z in state['z']])
            samples['loglik'].append(float(ll_cur if ll_cur is not None else calc_loglik_current(Y, state, opt)))
            samples['K'].append(int(diag.get('K', len(state['beta']))))
            samples['prune_info'].append(diag.get('prune', {}))

        if need_print:
            try:
                K_now = int(diag.get('K', len(state['beta'])))
                msg = f"[iHMM] iter {it+1}/{n_iter} | K={K_now}"
                if print_loglik and ll_cur is not None:
                    msg += f" | ll={ll_cur:.2f}"
                print(msg)
            except Exception:
                pass
    return state, samples
