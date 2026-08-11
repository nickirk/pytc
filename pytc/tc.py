"""Compatibility shim: pytc.tc is a TRUE ALIAS for pytc.integrals.tc (task #14,
isdf-coulomb-cuda, 2026-07-13) -- the canonical implementation moved to
pytc/integrals/tc.py as a peer of xtc.py/coulomb.py. This module and
pytc.integrals.tc are the SAME module object (sys.modules identity, not a
re-exported copy of names), verified via a standalone test before adoption:
`pytc.tc is pytc.integrals.tc` holds under cold import, from-import, and
package-eager-init, and `unittest.mock.patch("pytc.tc.<name>", ...)`
correctly mutates the value the real implementation's own functions read as
their global -- an explicit named re-export would NOT have this property
(mutating the shim's copy of a name leaves the real module's globals, and
thus its function bodies, untouched).
"""
import sys

from pytc.integrals import tc as _tc

sys.modules[__name__] = _tc
