# VMC (Variational Monte Carlo) module
# Re-export main functions for backwards compatibility

from .optimization import optimize_ref_var
from .mcmc_utils import (
    create_optimizer,
    create_gradient_mask
)

__all__ = [
    # Optimization functions
    'optimize_ref_var',
    # Utility functions
    'create_optimizer',
    'create_gradient_mask'
]
