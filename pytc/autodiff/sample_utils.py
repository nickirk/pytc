"""Utility functions for analyzing quantum Monte Carlo samples."""

import jax.numpy as jnp
from typing import Dict, Any, List, Optional, Tuple

def analyze_energies(sampling_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze energy convergence and statistics from sampling results.
    
    Args:
        sampling_results: Dictionary returned by metropolis_hastings
        
    Returns:
        Dictionary with energy statistics
    """
    energies = sampling_results["energies"]
    
    # Flatten the energies if they have walker dimension
    if len(energies.shape) > 1:
        flat_energies = energies.reshape(-1)
    else:
        flat_energies = energies
    
    # Calculate statistics
    energy_mean = jnp.mean(flat_energies)
    energy_error = jnp.std(flat_energies) / jnp.sqrt(len(flat_energies))
    energy_variance = jnp.var(flat_energies)
    
    # Calculate moving average
    window_size = 10
    cumsum = jnp.cumsum(jnp.insert(flat_energies, 0, 0))
    moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    
    
    # Calculate autocorrelation
    n = len(flat_energies)
    mean = jnp.mean(flat_energies)
    var = jnp.var(flat_energies)
    
    # Simple autocorrelation calculation for lag-1
    autocorr_1 = jnp.sum((flat_energies[:-1] - mean) * (flat_energies[1:] - mean)) / ((n-1) * var)
    
    stats = {
        "mean": energy_mean,
        "error": energy_error,
        "variance": energy_variance,
        "autocorr_lag1": autocorr_1,
    }
    
    return stats

def electron_density(sampling_results: Dict[str, Any], grid_points: int = 50, 
                     range_xyz: Optional[List[Tuple[float, float]]] = None) -> jnp.ndarray:
    """Calculate electron density from sampling results.
    
    Args:
        sampling_results: Dictionary returned by metropolis_hastings
        grid_points: Number of grid points in each dimension
        range_xyz: Optional list of (min, max) tuples for x, y, z dimensions
                   If None, automatically determined from samples
                   
    Returns:
        3D array representing electron density
    """
    samples = sampling_results["samples"]
    
    # Flatten all walker and step dimensions to get list of all electron positions
    all_electrons = samples.reshape(-1, samples.shape[-2], samples.shape[-1])
    all_positions = all_electrons.reshape(-1, all_electrons.shape[-1])
    
    # Determine grid range if not provided
    if range_xyz is None:
        min_xyz = jnp.min(all_positions, axis=0) - 1.0
        max_xyz = jnp.max(all_positions, axis=0) + 1.0
        range_xyz = [(min_xyz[i], max_xyz[i]) for i in range(3)]
    
    # Create grid
    x = jnp.linspace(range_xyz[0][0], range_xyz[0][1], grid_points)
    y = jnp.linspace(range_xyz[1][0], range_xyz[1][1], grid_points)
    z = jnp.linspace(range_xyz[2][0], range_xyz[2][1], grid_points)
    
    # Initialize density array
    density = jnp.zeros((grid_points, grid_points, grid_points))
    
    # Simple histogram approach for electron density
    x_indices = jnp.clip(jnp.floor((all_positions[:, 0] - range_xyz[0][0]) / 
                          (range_xyz[0][1] - range_xyz[0][0]) * grid_points).astype(jnp.int32), 
                          0, grid_points-1)
    y_indices = jnp.clip(jnp.floor((all_positions[:, 1] - range_xyz[1][0]) / 
                          (range_xyz[1][1] - range_xyz[1][0]) * grid_points).astype(jnp.int32), 
                          0, grid_points-1)
    z_indices = jnp.clip(jnp.floor((all_positions[:, 2] - range_xyz[2][0]) / 
                          (range_xyz[2][1] - range_xyz[2][0]) * grid_points).astype(jnp.int32), 
                          0, grid_points-1)
    
    # Use numpy for histogram generation since jax doesn't have an equivalent
    import numpy as np
    density_np = np.zeros((grid_points, grid_points, grid_points))
    
    # Convert JAX arrays to numpy
    x_indices_np = np.array(x_indices)
    y_indices_np = np.array(y_indices)
    z_indices_np = np.array(z_indices)
    
    # Count electrons in each grid cell
    for i in range(len(x_indices_np)):
        density_np[x_indices_np[i], y_indices_np[i], z_indices_np[i]] += 1
    
    # Normalize
    density = jnp.array(density_np) / len(all_positions)
    
    return density, (x, y, z)

def plot_density_2d(density, grid, plane='xy', index=None):
    """Plot a 2D slice of the electron density.
    
    Args:
        density: 3D array from electron_density function
        grid: Tuple of (x, y, z) grid arrays
        plane: Which plane to plot ('xy', 'xz', or 'yz')
        index: Index for the third dimension, if None uses middle of grid
    """
    import matplotlib.pyplot as plt

    x, y, z = grid
    
    if plane == 'xy':
        if index is None:
            index = len(z) // 2
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(x, y, density[:, :, index].T, shading='auto')
        plt.colorbar(label='Electron Density')
        plt.xlabel('x (bohr)')
        plt.ylabel('y (bohr)')
        plt.title(f'Electron Density in xy-plane at z={z[index]:.2f}')
    elif plane == 'xz':
        if index is None:
            index = len(y) // 2
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(x, z, density[:, index, :].T, shading='auto')
        plt.colorbar(label='Electron Density')
        plt.xlabel('x (bohr)')
        plt.ylabel('z (bohr)')
        plt.title(f'Electron Density in xz-plane at y={y[index]:.2f}')
    elif plane == 'yz':
        if index is None:
            index = len(x) // 2
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(y, z, density[index, :, :].T, shading='auto')
        plt.colorbar(label='Electron Density')
        plt.xlabel('y (bohr)')
        plt.ylabel('z (bohr)')
        plt.title(f'Electron Density in yz-plane at x={x[index]:.2f}')
    
    plt.tight_layout()
    plt.show()
