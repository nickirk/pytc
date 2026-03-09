import numpy as np
import logging
import jax.tree_util as jtu
from typing import Dict, Any, Union, Tuple
from .mcmc_utils import load_optimization_history

logger = logging.getLogger(__name__)

def block_analysis(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform block analysis on a 1D array of correlated data to estimate standard error.
    
    Args:
        data: 1D numpy array of correlated samples (e.g. energies from MCMC).
        
    Returns:
        A tuple of (block_sizes, block_errors, block_overall_means):
        - block_sizes: Array of block sizes used.
        - block_errors: Array of estimated standard errors for each block size.
        - block_overall_means: Array of the mean value of the data for the given block size.
    """
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("Data must be a 1D array for standard block analysis.")
        
    n_data = len(data)
    max_blocks = int(np.log2(n_data))
    
    if max_blocks < 1:
        return np.array([1]), np.array([0.0]), np.array([np.mean(data) if n_data > 0 else 0.0])
        
    block_sizes = []
    block_errors = []
    block_overall_means = []
    
    for i in range(max_blocks):
        block_size = 2**i
        n_blocks = n_data // block_size
        
        # Truncate data to neatly fit into blocks
        truncated_data = data[:n_blocks * block_size]
        blocks = truncated_data.reshape((n_blocks, block_size))
        
        # Mean of each block
        block_means = np.mean(blocks, axis=1)
        
        if n_blocks > 1:
            # Error of the means
            block_var = np.var(block_means, ddof=1)
            block_error = np.sqrt(block_var / n_blocks)
            
            block_sizes.append(block_size)
            block_errors.append(block_error)
            block_overall_means.append(np.mean(block_means))
            
    return np.array(block_sizes), np.array(block_errors), np.array(block_overall_means)

def detect_equilibration_cma(data: np.ndarray, threshold_sigma: float = 0.5, print_results: bool = True) -> int:
    """
    Detects the equilibration point (burn-in length) using the reverse 
    Cumulative Moving Average (CMA) deviation method.
    
    It scans backward from the end of the simulation. For each step i, it calculates
    the mean of the data from i to the end. The burn-in period is considered over when 
    this 'tail mean' is within a tight threshold of the final steady-state mean.
    """
    data = np.asarray(data)
    n = len(data)
    if n < 100:
        return 0
        
    # Assume the second half of the simulation is firmly equilibrated to define the steady state
    steady_state_tail = data[n // 2:]
    steady_state_mean = np.mean(steady_state_tail)
    steady_state_std = np.std(steady_state_tail)
    
    # Calculate reverse CMA: cma_from_end[i] is the mean of data[i:n]
    rev_data = data[::-1]
    rev_cma = np.cumsum(rev_data) / np.arange(1, n + 1)
    cma_from_end = rev_cma[::-1]
    
    deviations = np.abs(cma_from_end - steady_state_mean)
    
    # We use a strict threshold based on the intrinsic noise of the data
    threshold = threshold_sigma * steady_state_std
    
    if print_results:
        logger.info(f"Reverse CMA: Assumed steady state is the last {len(steady_state_tail)} iterations.")
        logger.info(f"Reverse CMA: Steady state mean = {steady_state_mean:.6g}, deviation std = {steady_state_std:.6g}")
        logger.info(f"Reverse CMA: Tolerance threshold for burn-in = {threshold:.6g} ({threshold_sigma} sigma)")
    
    exceeds_threshold = deviations > threshold
    
    if not np.any(exceeds_threshold):
        if print_results:
            logger.info("Reverse CMA: Entire trace appears within tolerance. No burn-in detected.")
        return 0
        
    # Find the last index where the tail mean was still biased by initial configurations
    burn_in_idx = np.where(exceeds_threshold)[0][-1] + 1
    
    if print_results:
        logger.info(f"Reverse CMA: Trace deviates significantly from steady state mean until step {burn_in_idx}.")
    
    # Cap burn-in at 80% to ensure we always have at least some data to analyze
    return min(burn_in_idx, int(0.8 * n))

def analyze_optimization_history(
    history: Union[str, Dict[str, Any]], 
    burn_in_frac: Union[float, str] = 'auto',
    print_results: bool = True
) -> Dict[str, Any]:
    """
    Analyze an optimization history, discarding burn-in and computing block analysis.
    
    Args:
        history: Filepath to an HDF5 history file, or a dictionary containing history data.
        burn_in_frac: Fraction of the initial steps to discard as burn-in.
        print_results: Whether to print textual output.
        
    Returns:
        A dictionary containing the block analysis results and optimal parameters (Polyak averaged).
    """
    if isinstance(history, str):
        if print_results:
            logger.info(f"Loading history from {history}...")
        history = load_optimization_history(history)
        
    energies = np.array(history['energies'])
    n_total = len(energies)
    
    if isinstance(burn_in_frac, str) and burn_in_frac.lower() == 'auto':
        burn_in_idx = detect_equilibration_cma(energies, print_results=print_results)
        if print_results:
            logger.info(f"Auto-detected burn-in point at step {burn_in_idx} using Reverse CMA method.")
    else:
        burn_in_idx = int(burn_in_frac * n_total)
        
    if n_total < 100:
        burn_in_idx = 0
         
    equil_energies = energies[burn_in_idx:]
    
    if print_results:
        logger.info(f"Loaded {n_total} energy values.")
        logger.info(f"Using {len(equil_energies)} energies after discarding {burn_in_idx} for burn-in.")
        
    mean_energy = np.mean(equil_energies)
    
    sizes, errors, block_means_arr = block_analysis(equil_energies)
    max_err = np.max(errors) if len(errors) > 0 else 0.0
    
    if print_results:
        logger.info(f"Mean Energy: {mean_energy:.6f}")
        logger.info("\\nBlock Analysis:")
        logger.info(f"{'Block Size':>12} {'Mean Energy':>16} {'Std Error':>15}")
        for s, m, e in zip(sizes, block_means_arr, errors):
            logger.info(f"{s:12d} {m:16.6f} {e:15.6f}")
            
        logger.info(f"\\nEstimated standard error (from plateau): {max_err:.6f}")
        logger.info(f"Final Result:         {mean_energy:.6f} +/- {max_err:.6f}")
        
    # Also perform Polyak-Ruppert averaging on the parameters if they exist
    polyak_params = None
    if 'params' in history:
        params_history = history['params']
        # Extract the equilibrated part of the parameters history (slices each leaf's axis=0)
        equil_params_history = jtu.tree_map(lambda x: x[burn_in_idx:], params_history)
        
        # Average each parameter over the equilibrated trace
        polyak_params = jtu.tree_map(lambda x: np.mean(x, axis=0), equil_params_history)
            
    return {
        "mean_energy": mean_energy,
        "standard_error": max_err,
        "block_sizes": sizes,
        "block_errors": errors,
        "block_means": block_means_arr,
        "polyak_params": polyak_params
    }
