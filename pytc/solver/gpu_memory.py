"""Backward-compatible re-export — module moved to pytc.utils.gpu_memory."""
from pytc.utils.gpu_memory import *  # noqa: F401,F403
from pytc.utils.gpu_memory import (  # explicit names for IDE support
    estimate_persistent_gpu_bytes,
    get_gpu_budget_bytes,
    _get_gpu_physical_bytes,
    _get_gpu_free_bytes,
    estimate_blksize,
    adaptive_rank_block_size,
    enable_xla_compilation_cache,
)
