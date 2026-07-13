"""Compatibility shim: pytc.xtc is a TRUE ALIAS for pytc.integrals.xtc (task
#14, isdf-coulomb-cuda, 2026-07-13) -- the canonical implementation moved to
pytc/integrals/xtc.py as a peer of tc.py/coulomb.py. This module and
pytc.integrals.xtc are the SAME module object (sys.modules identity, not a
re-exported copy of names) -- see pytc/tc.py's docstring for the verification
this pattern was tested against before adoption, including the
`unittest.mock.patch("pytc.xtc.<name>", ...)` case several existing tests
rely on.
"""
import sys

from pytc.integrals import xtc as _xtc

sys.modules[__name__] = _xtc
