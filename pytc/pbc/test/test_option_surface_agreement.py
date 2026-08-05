"""build() and ISDFDF.__init__() expose one option surface, maintained twice.

The two carry ~20 shared parameters with hand-written duplicate defaults. A
parameter added to one and not the other, or a default changed on one side,
diverges silently: both spellings keep working and produce different builds.
"""

import inspect
import unittest

from pytc.pbc import coulomb


def _params(fn):
    sig = inspect.signature(fn)
    return {
        name: p for name, p in sig.parameters.items()
        if name not in ("self", "cls")
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }


#: Build-time only by design: a callback has no meaning as adapter state.
BUILD_ONLY = {"on_selection"}
INIT_ONLY = set()

#: Divergences that exist today and have NOT been ruled deliberate. Listed so
#: the suite passes on the current tree while keeping them visible; each needs
#: an owner decision to be promoted to BUILD_ONLY/INIT_ONLY or closed by adding
#: the parameter to the other side.
#:   provider_cls -- build() callers may choose the kernel provider; callers
#:   going through the ISDFDF adapter always get the default.
UNREVIEWED_DIVERGENCE = {"provider_cls"}


class TestOptionSurfaceAgreement(unittest.TestCase):
    def setUp(self):
        self.build = _params(coulomb.build)
        self.init = _params(coulomb.ISDFDF.__init__)

    def test_name_sets_agree(self):
        # Checked before defaults: a parameter added to one side only would
        # otherwise pass a shared-defaults comparison unnoticed.
        build_extra = set(self.build) - set(self.init) - BUILD_ONLY - UNREVIEWED_DIVERGENCE
        init_extra = set(self.init) - set(self.build) - INIT_ONLY - UNREVIEWED_DIVERGENCE
        self.assertEqual(
            (build_extra, init_extra), (set(), set()),
            msg=(
                "build() and ISDFDF.__init__() option names have diverged. "
                f"only in build(): {sorted(build_extra)}; "
                f"only in __init__(): {sorted(init_extra)}. "
                "Add it to both, or list it in BUILD_ONLY/INIT_ONLY with a reason."
            ),
        )

    def test_shared_defaults_agree(self):
        shared = (set(self.build) & set(self.init)) - BUILD_ONLY - INIT_ONLY - UNREVIEWED_DIVERGENCE
        mismatched = {
            name: (self.build[name].default, self.init[name].default)
            for name in sorted(shared)
            if self.build[name].default is not inspect.Parameter.empty
            and self.init[name].default is not inspect.Parameter.empty
            and self.build[name].default != self.init[name].default
        }
        self.assertEqual(
            mismatched, {},
            msg=(
                "build() and ISDFDF.__init__() disagree on default values "
                f"(name: (build, __init__)): {mismatched}"
            ),
        )

    def test_required_ness_agrees(self):
        # A parameter optional on one side and required on the other is a
        # divergence the two checks above both miss.
        shared = (set(self.build) & set(self.init)) - BUILD_ONLY - INIT_ONLY - UNREVIEWED_DIVERGENCE
        mismatched = sorted(
            name for name in shared
            if (self.build[name].default is inspect.Parameter.empty)
            != (self.init[name].default is inspect.Parameter.empty)
        )
        self.assertEqual(mismatched, [], msg=f"required-ness differs for {mismatched}")

    def test_the_surface_is_actually_shared(self):
        # Guards the guard: if the two stopped overlapping, the checks above
        # would pass vacuously over an empty set.
        shared = (set(self.build) & set(self.init)) - BUILD_ONLY - INIT_ONLY - UNREVIEWED_DIVERGENCE
        self.assertGreaterEqual(
            len(shared), 15,
            msg=f"expected a large shared option surface, found {len(shared)}",
        )

    def test_carve_outs_are_real(self):
        # A stale carve-out would mask a genuine divergence forever.
        for name in UNREVIEWED_DIVERGENCE:
            in_build, in_init = name in self.build, name in self.init
            self.assertNotEqual(
                in_build, in_init,
                f"{name!r} no longer diverges; remove it from UNREVIEWED_DIVERGENCE",
            )
        for name in BUILD_ONLY:
            self.assertIn(name, self.build, f"BUILD_ONLY lists {name!r}, not in build()")
            self.assertNotIn(name, self.init, f"{name!r} is in __init__(); drop the carve-out")
        for name in INIT_ONLY:
            self.assertIn(name, self.init, f"INIT_ONLY lists {name!r}, not in __init__()")
            self.assertNotIn(name, self.build, f"{name!r} is in build(); drop the carve-out")


if __name__ == "__main__":
    unittest.main()
