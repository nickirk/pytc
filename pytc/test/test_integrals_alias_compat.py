"""Regression test for the pytc.tc/pytc.xtc -> pytc/integrals/ module-alias
shim (task #14, isdf-coulomb-cuda, commit A corrective follow-up, 2026-07-13).

Alice's review of commit A (cb05dff) found the sys.modules-aliasing identity
and patch-through behavior had only been verified via a throwaway/manual
experiment, not a committed test. This makes that verification durable:

- ``pytc.tc``/``pytc.xtc`` are the literal same module object as
  ``pytc.integrals.tc``/``pytc.integrals.xtc`` (sys.modules identity), not an
  explicit-re-export copy.
- root ``pytc.integrals`` is an explicit, documented package-surface member
  (``pytc.__all__``), not incidentally present because a shim imported it.
- ``unittest.mock.patch("pytc.tc.<name>", ...)``-style patching through the
  shim path mutates what the canonical module's own functions read as their
  global, since it is the same module dict.
- ``importlib.reload`` through the shim name does not desynchronize the
  alias.
- ``import pytc`` does not eagerly import ``pytc.integrals.coulomb`` (kept
  lazy so its heavier pyscf/gpu4pyscf-adjacent dependencies only load on
  demand).
"""
import importlib
import json
import subprocess
import sys
import unittest
from unittest import mock


class TestIntegralsAliasIdentity(unittest.TestCase):
    def test_tc_alias_is_canonical_module(self):
        import pytc
        import pytc.integrals.tc
        self.assertIs(pytc.tc, pytc.integrals.tc)
        self.assertIs(pytc.tc.__dict__, pytc.integrals.tc.__dict__)
        self.assertEqual(pytc.tc.__name__, "pytc.integrals.tc")

    def test_xtc_alias_is_canonical_module(self):
        import pytc
        import pytc.integrals.xtc
        self.assertIs(pytc.xtc, pytc.integrals.xtc)
        self.assertIs(pytc.xtc.__dict__, pytc.integrals.xtc.__dict__)
        self.assertEqual(pytc.xtc.__name__, "pytc.integrals.xtc")

    def test_integrals_package_exposed_on_root(self):
        import pytc
        self.assertIn("integrals", pytc.__all__)
        self.assertIs(pytc.integrals, sys.modules["pytc.integrals"])


class TestPatchThroughShim(unittest.TestCase):
    """Confirms mock.patch("pytc.tc.<name>", ...) / pytc.xtc.<name> mutate
    what the real implementation's own functions read as their global --
    the exact property an explicit-re-export shim would NOT have, and the
    reason several existing tests' mock.patch targets still work correctly.
    """

    def test_patch_pytc_tc_name_visible_in_canonical_module_and_function_globals(self):
        import pytc.tc
        import pytc.integrals.tc as canonical

        with mock.patch("pytc.tc.logger", "sentinel-value"):
            self.assertEqual(canonical.logger, "sentinel-value")
            # A real function defined in tc.py resolves its unqualified
            # `logger` reference via its own __globals__ -- confirm that
            # dict is the exact one the patch mutated.
            self.assertIs(canonical.TC.get_2b.__globals__, pytc.tc.__dict__)
        self.assertNotEqual(canonical.logger, "sentinel-value")

    def test_patch_pytc_xtc_name_visible_in_canonical_module_and_function_globals(self):
        import pytc.xtc
        import pytc.integrals.xtc as canonical

        with mock.patch("pytc.xtc.logger", "sentinel-value"):
            self.assertEqual(canonical.logger, "sentinel-value")
            self.assertIs(canonical.XTC.get_2b.__globals__, pytc.xtc.__dict__)
        self.assertNotEqual(canonical.logger, "sentinel-value")


class TestReloadPreservesAlias(unittest.TestCase):
    def test_reload_via_shim_name_keeps_identity(self):
        import pytc.tc
        import pytc.integrals.tc

        importlib.reload(pytc.tc)

        # Re-fetch after reload -- both names must still resolve to one
        # object, and the shim's own sys.modules entry must not have been
        # replaced by a stale pre-reload reference.
        import pytc.tc as tc_after
        import pytc.integrals.tc as canonical_after
        self.assertIs(tc_after, canonical_after)
        self.assertIs(sys.modules["pytc.tc"], sys.modules["pytc.integrals.tc"])

    def test_reload_via_xtc_shim_name_keeps_identity(self):
        import pytc.xtc
        import pytc.integrals.xtc

        importlib.reload(pytc.xtc)

        import pytc.xtc as xtc_after
        import pytc.integrals.xtc as canonical_after
        self.assertIs(xtc_after, canonical_after)
        self.assertIs(sys.modules["pytc.xtc"], sys.modules["pytc.integrals.xtc"])


class TestNoEagerCoulombImport(unittest.TestCase):
    """`import pytc` must not eagerly pull in pytc.integrals.coulomb (kept
    lazy, per pytc/integrals/__init__.py's own docstring). Run in a fresh
    subprocess so an already-populated sys.modules from other tests in this
    process cannot mask an eager-import regression.
    """

    def test_fresh_import_pytc_does_not_load_coulomb(self):
        r = subprocess.run(
            [sys.executable, "-c",
             "import pytc, sys, json; "
             "print('RESULT ' + json.dumps("
             "{'has_coulomb': 'pytc.integrals.coulomb' in sys.modules}))"],
            capture_output=True, text=True, timeout=300,
        )
        if r.returncode != 0:
            self.fail(f"child import failed:\n--- stdout ---\n{r.stdout}\n"
                      f"--- stderr ---\n{r.stderr}")
        for line in r.stdout.splitlines():
            if line.startswith("RESULT "):
                result = json.loads(line[len("RESULT "):])
                self.assertFalse(
                    result["has_coulomb"],
                    "import pytc must not eagerly import pytc.integrals.coulomb",
                )
                return
        self.fail(f"no RESULT line in child stdout:\n{r.stdout}")


if __name__ == "__main__":
    unittest.main()
