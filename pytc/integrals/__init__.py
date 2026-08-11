"""Peer integral-model modules (tc, xtc, coulomb) -- task #14, isdf-coulomb-cuda,
2026-07-13: canonical implementations live here, composing the model-agnostic
fitting machinery in pytc.df. Root pytc.tc/pytc.xtc remain importable as true
module aliases (sys.modules identity, not a re-export copy) for established
external callers.

Deliberately lightweight: no eager imports of tc/xtc/coulomb here (coulomb
pulls in pyscf/gpu4pyscf-heavy dependencies that should only load when
actually requested, not on every `import pytc`).
"""
