"""Logging utilities for pytc.

This module provides centralized logging functionality including:
- Git commit information for reproducibility
- Dependency version tracking
- Device detection and information
"""

import logging
import subprocess
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def get_git_commit(repo_path: Optional[Path] = None) -> str:
    """Get the current git commit hash.

    Args:
        repo_path: Path to the git repository. Defaults to the pytc package directory.

    Returns:
        The git commit hash (short form), or 'unknown' if not available.
    """
    if repo_path is None:
        repo_path = Path(__file__).parent.parent

    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        commit_hash = result.stdout.strip()
        return commit_hash[:7]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


def get_pytc_version() -> str:
    """Get the pytc version from pyproject.toml.

    Returns:
        The pytc version string, or 'unknown' if not available.
    """
    try:
        from importlib.metadata import version
        return version('pytc-qc')
    except PackageNotFoundError:
        pass

    pyproject_path = Path(__file__).parent.parent / 'pyproject.toml'
    if pyproject_path.exists():
        import tomllib
        with open(pyproject_path, 'rb') as f:
            data = tomllib.load(f)
            return data.get('project', {}).get('version', 'unknown')
    return 'unknown'


def get_dependency_versions() -> Dict[str, str]:
    """Get versions of key dependencies.

    Returns:
        Dictionary mapping package names to their versions.
    """
    deps = {}

    packages = [
        'jax',
        'numpy',
        'optax',
        'flax',
        'pyscf',
        'folx',
    ]

    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            deps[pkg] = version
        except ImportError:
            deps[pkg] = 'not installed'

    return deps


def get_device_info() -> Dict[str, any]:
    """Get information about available compute devices (GPUs/CPUs).

    Returns:
        Dictionary with device information including type, count, and memory.
    """
    info = {
        'device_type': 'unknown',
        'device_count': 0,
        'devices': [],
    }

    try:
        import jax
        devices = jax.devices()

        info['device_count'] = len(devices)

        if devices:
            device_kind = devices[0].device_kind
            info['device_type'] = device_kind.lower()

            for dev in devices:
                dev_info = {
                    'id': dev.id,
                    'kind': dev.device_kind,
                }
                try:
                    stats = dev.memory_stats()
                    if stats:
                        total_bytes = stats.get('bytes_in_use', 0) + stats.get('bytes_available', 0)
                        dev_info['total_memory_gb'] = total_bytes / (1024**3)
                except Exception:
                    pass
                info['devices'].append(dev_info)

    except ImportError:
        info['device_type'] = 'jax not available'
    except Exception as e:
        info['device_type'] = f'error: {e}'

    return info


def log_startup_info():
    """Log startup information for reproducibility.

    This logs:
    - pytc version
    - git commit hash
    - dependency versions
    - device information
    """
    pytc_version = get_pytc_version()
    git_commit = get_git_commit()
    deps = get_dependency_versions()
    device_info = get_device_info()

    logger.info("=" * 60)
    logger.info("pytc startup information")
    logger.info("=" * 60)
    logger.info(f"pytc version: {pytc_version}")
    logger.info(f"git commit: {git_commit}")

    logger.info("Dependency versions:")
    for pkg, version in deps.items():
        logger.info(f"  {pkg}: {version}")

    logger.info("Device information:")
    logger.info(f"  Device type: {device_info['device_type']}")
    logger.info(f"  Device count: {device_info['device_count']}")
    for dev in device_info['devices']:
        mem_str = ""
        if 'total_memory_gb' in dev:
            mem_str = f", {dev['total_memory_gb']:.1f} GB"
        logger.info(f"    Device {dev['id']}: {dev['kind']}{mem_str}")

    logger.info("=" * 60)
