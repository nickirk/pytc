import folx.ad
import jax
import jax.numpy as jnp
import jax.flatten_util as jfu

def patched_jacrev(f):
    # Similar to jax.jacrev but works with complex inputs and outputs.
    # A crucial difference is that we do not preserve the structure of the output but
    # always flatten it to a 1D array.
    def jacfun(*primals):
        flat_primals, unravel = jfu.ravel_pytree(primals)

        def flat_f(x):
            return jfu.ravel_pytree(f(*unravel(x)))[0]

        out = flat_f(flat_primals)
        
        eye = jnp.eye(out.size, dtype=out.dtype)
        
        # Patch for multi-GPU/shard_map support:
        # If 'out' is a ShardMapTracer (or similar), we must ensure 'eye' 
        # carries the same sharding/tracer info so that the VJP accepts it.
        # We achieve this by adding a zeroed version of 'out' broadcasted to shape.
        if hasattr(out, 'aval'): 
             eye = eye + jnp.zeros_like(out)[None, :]

        result = jax.vmap(folx.ad.vjp(flat_f, flat_primals))(
            eye
        )[0]
        result = jax.vmap(unravel, out_axes=0)(result)
        if len(primals) == 1:
            return result[0]
        return result

    return jacfun

def apply_fix():
    """Apply monkeypatch to folx.ad.jacrev to fix sharding compatibility."""
    folx.ad.jacrev = patched_jacrev
