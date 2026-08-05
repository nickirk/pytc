"""build() and ISDFDF.__init__() expose the same options, maintained twice."""

import inspect
import unittest

from pytc.pbc import coulomb

# on_selection: a build-time callback, meaningless as adapter state.
# provider_cls: reachable through build() but not the adapter. Unreviewed --
#   listed here to keep the suite honest, not because it is known to be right.
BUILD_ONLY = {"on_selection", "provider_cls"}


def _params(fn):
    return {
        name: p for name, p in inspect.signature(fn).parameters.items()
        if name not in ("self", "cls")
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }


class TestOptionSurfaceAgreement(unittest.TestCase):
    def setUp(self):
        self.build = _params(coulomb.build)
        self.init = _params(coulomb.ISDFDF.__init__)

    def test_option_names_agree(self):
        self.assertEqual(set(self.build) - set(self.init), BUILD_ONLY)
        self.assertEqual(set(self.init) - set(self.build), set())

    def test_shared_defaults_agree(self):
        shared = (set(self.build) & set(self.init)) - BUILD_ONLY
        self.assertGreaterEqual(len(shared), 15, "shared surface vanished; check is vacuous")
        for name in sorted(shared):
            with self.subTest(option=name):
                self.assertEqual(self.build[name].default, self.init[name].default)


if __name__ == "__main__":
    unittest.main()
