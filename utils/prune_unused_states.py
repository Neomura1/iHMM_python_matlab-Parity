"""Prune states with usage below a threshold and remap labels.

MATLAB counterpart: prune_unused_states.m
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def prune_unused_states(state: Dict, threshold: int = 0) -> Tuple[Dict, Dict]:
    """Prune states whose total occupancy is <= threshold.

    Parameters
    ----------
    state : dict
        iHMM state with keys at least 'z', 'theta', 'Pi', 'beta'.
    threshold : int, default 0
        Minimum number of assigned observations required to keep a state.

    Returns
    -------
    state : dict
        Updated state with pruned/renumbered components.
    info : dict
        Diagnostics including 'keep_mask', 'occ', 'map_old_to_new', 'K_old', 'K_new'.
    """

    if 'z' not in state:
        return state, {
            'keep_mask': np.array([], dtype=bool),
            'occ': np.array([], dtype=int),
            'map_old_to_new': np.array([], dtype=int),
            'K_old': 0,
            'K_new': 0,
        }

    # Derive K robustly from available fields
    K_candidates = []
    if 'beta' in state:
        K_candidates.append(len(state['beta']))
    if 'theta' in state:
        K_candidates.append(len(state['theta']))
    if 'Pi' in state:
        K_candidates.append(state['Pi'].shape[0])
    K = int(max(K_candidates) if K_candidates else 0)
    if K == 0:
        return state, {
            'keep_mask': np.array([], dtype=bool),
            'occ': np.array([], dtype=int),
            'map_old_to_new': np.array([], dtype=int),
            'K_old': 0,
            'K_new': 0,
        }

    # Compute occupancy across sequences
    occ = np.zeros(K, dtype=int)
    for z in state['z']:
        occ += np.bincount(np.asarray(z, dtype=int), minlength=K)

    keep = occ > threshold
    K_new = int(keep.sum())

    if K_new == K:
        return state, {
            'keep_mask': keep,
            'occ': occ,
            'map_old_to_new': np.arange(K),
            'K_old': K,
            'K_new': K,
        }

    if K_new == 0:
        # Keep at least the most frequent (or 0 if all zero)
        keep = np.zeros(K, dtype=bool)
        if np.any(occ > 0):
            keep[int(np.argmax(occ))] = True
        else:
            keep[0] = True
        K_new = 1

    # Build index mapping old->new
    new_idx = -np.ones(K, dtype=int)
    new_idx[np.where(keep)[0]] = np.arange(K_new)

    # Remap z
    z_list_new = [new_idx[np.asarray(z, dtype=int)] for z in state['z']]
    state['z'] = z_list_new

    # Prune parameters accordingly
    if 'theta' in state and isinstance(state['theta'], list):
        state['theta'] = [th for th, kf in zip(state['theta'], keep) if kf]
    if 'Pi' in state and getattr(state['Pi'], 'size', 0):
        state['Pi'] = state['Pi'][keep][:, keep]
    if 'beta' in state and len(state['beta']):
        beta = state['beta'][keep]
        total = float(beta.sum())
        state['beta'] = beta / total if total > 0 else np.ones(K_new) / K_new

    # Optionally align counts/tables if present (best-effort)
    if 'counts' in state and getattr(state['counts'], 'size', 0):
        try:
            state['counts'] = state['counts'][keep][:, keep]
        except Exception:
            pass
    if 'tables' in state and getattr(state['tables'], 'size', 0):
        try:
            state['tables'] = state['tables'][keep][:, keep]
        except Exception:
            pass

    info = {
        'keep_mask': keep,
        'occ': occ,
        'map_old_to_new': new_idx,
        'K_old': K,
        'K_new': K_new,
    }
    return state, info

