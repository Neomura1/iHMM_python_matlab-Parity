"""State birth/death/prune utilities.

This implementation provides pruning of unused states during sampling. Birth is
left as a future extension and can be added based on split-merge proposals or
slice-based activation.
"""

from __future__ import annotations

from typing import Dict, Tuple

from utils.prune_unused_states import prune_unused_states


def birth_death_prune(state: Dict, opt: Dict) -> Tuple[Dict, Dict]:
    """Apply pruning (and optionally birth) to the current state.

    Parameters
    ----------
    state : dict
        Current iHMM state; must include 'z', 'theta', 'Pi', 'beta'.
    opt : dict
        Options; recognized keys:
          - 'prune_unused_states' (bool): enable occupancy-based pruning (default True)
          - 'prune_min_count' (int): minimum occupancy required to keep a state (default 0)

    Returns
    -------
    state : dict
        Potentially modified state.
    info : dict
        Diagnostics from pruning (and birth if added later).
    """

    info = {}
    if opt.get('prune_unused_states', True):
        min_count = int(opt.get('prune_min_count', 0))
        state, info = prune_unused_states(state, threshold=min_count)

    # Placeholder for birth moves (not implemented):
    # if opt.get('enable_birth', False):
    #     state, birth_info = propose_birth(state, opt)
    #     info.update({f"birth_{k}": v for k, v in birth_info.items()})

    return state, info

