# VMC (Variational Monte Carlo) module
# Re-export main functions for backwards compatibility

from .walker import Walker, initialize_walker_state, initialize_walkers
from .moves import _all_electron_move, _one_electron_move, _compute_green_function
from .metropolis import metropolis_hastings, metropolis_hastings_importance_sampling
from .sampling import burn_in, burn_in_with_importance, sample
from .optimization import optimize, optimize_ref_var, evaluate_ref_var
from .mcmc_utils import (
    prepare_sampling_results, report_progress,
    init_electron_configs, save_walkers, load_walkers, resample_walkers
)
from .optimizer import create_optimizer, create_gradient_mask
from .blocking import block_analysis, analyze_optimization_history
from .sharding import (
    create_mesh, shard_walker, replicate,
    pad_n_walkers, pad_walker,
    get_walker_sharding, get_replicated_sharding,
    get_vmap_fn, is_multi_gpu, n_devices,
)

__all__ = [
    # Walker functions
    'Walker', 'initialize_walker_state', 'initialize_walkers',
    # Move functions  
    '_all_electron_move', '_one_electron_move', '_compute_green_function',
    # Metropolis algorithms
    'metropolis_hastings', 'metropolis_hastings_importance_sampling',
    # Sampling functions
    'burn_in', 'burn_in_with_importance', 'sample',
    # Optimization functions
    'optimize', 'optimize_ref_var', 'evaluate_ref_var',
    # Utility functions
    'prepare_sampling_results', 'report_progress', 'create_optimizer',
    'init_electron_configs', 'create_gradient_mask',
    'save_walkers', 'load_walkers', 'resample_walkers',
    'block_analysis', 'analyze_optimization_history',
    # Sharding utilities
    'create_mesh', 'shard_walker', 'replicate',
    'pad_n_walkers', 'pad_walker',
    'get_walker_sharding', 'get_replicated_sharding',
    'get_vmap_fn', 'is_multi_gpu', 'n_devices',
]
