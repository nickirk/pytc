"""Import-compatibility test for the pytc/df.py -> pytc/df/ package
reorganization (task #8, isdf-coulomb-cuda, 2026-07-12).

Verifies every symbol previously importable as ``pytc.df.X`` when this
was a single module stays importable the same way after the split into
pytc/df/{__init__,pivots,solvers,isdf}.py -- Alice's migration caution:
"add an import-compatibility test for the currently used public
symbols." Covers every symbol found imported from pytc.df anywhere in
the codebase (grep-verified, 2026-07-12) plus the module's own public
surface.
"""
import unittest


class TestDfPackageImportCompat(unittest.TestCase):
    def test_public_symbols_importable_from_pytc_df(self):
        from pytc.df import (
            isdf_decompose,
            pivoted_cholesky_pair_pivots,
            prepare_spd_cholesky,
            prepare_normal_equations_solver,
            solve_normal_equations_batch_prepared,
            solve_normal_equations_batch,
        )
        for name, obj in [
            ("isdf_decompose", isdf_decompose),
            ("pivoted_cholesky_pair_pivots", pivoted_cholesky_pair_pivots),
            ("prepare_spd_cholesky", prepare_spd_cholesky),
            ("prepare_normal_equations_solver", prepare_normal_equations_solver),
            ("solve_normal_equations_batch_prepared", solve_normal_equations_batch_prepared),
            ("solve_normal_equations_batch", solve_normal_equations_batch),
        ]:
            self.assertTrue(callable(obj), f"pytc.df.{name} must remain callable")

    def test_underscore_symbol_used_externally_stays_importable(self):
        # _pivoted_cholesky_phi is underscore-prefixed but imported
        # directly by pytc/test/test_pivot_selection.py -- an
        # explicit exception to the private-name convention (Alice's
        # migration spec: "except re-exporting a symbol if an existing
        # import requires it").
        from pytc.df import _pivoted_cholesky_phi
        self.assertTrue(callable(_pivoted_cholesky_phi))

    def test_module_layout_matches_agreed_spec(self):
        # pytc/df/ must contain ONLY model-agnostic machinery -- no
        # tc.py/coulomb.py inside df/ (Ke/Alice/Felix's ratified
        # amendment, 2026-07-12: those responsibilities belong to a
        # future peer pytc/integrals/ layer, not inside df/).
        import pytc.df as df_pkg
        import pytc.df.pivots  # noqa: F401
        import pytc.df.solvers  # noqa: F401
        import pytc.df.isdf  # noqa: F401
        with self.assertRaises(ModuleNotFoundError):
            import pytc.df.tc  # noqa: F401
        with self.assertRaises(ModuleNotFoundError):
            import pytc.df.coulomb  # noqa: F401

    def test_coulomb_path_imports_from_new_df_package(self):
        # pytc/integrals/coulomb.py imports shared primitives from the
        # df/ package -- these must keep working unchanged (no
        # relocation of Coulomb's own business logic in task #8, per
        # its ratified narrow scope).
        from pytc.integrals.coulomb import select_sector_pivots  # noqa: F401
        from pytc.integrals.coulomb import compute_Z  # noqa: F401

    def test_module_attribute_access_pattern_from_tc_and_xtc_py(self):
        # pytc/tc.py AND pytc/xtc.py's actual production call sites both
        # use `from . import df` then `df.isdf_decompose(...)` (module-
        # attribute access, not a name import) -- packages resolve this
        # identically to modules via __init__.py's re-exports, but this
        # is exactly the pattern that would break silently if
        # __init__.py were ever missing a re-export.
        from pytc import df
        self.assertTrue(callable(df.isdf_decompose))

    def test_pytc_top_level_import_loads_df_package_cleanly(self):
        # pytc/__init__.py itself does `from . import df` at PACKAGE
        # LOAD TIME (not function-scoped like tc.py/xtc.py) -- a single
        # broken import anywhere in the new df/ package would break
        # `import pytc` entirely, for every caller, immediately.
        import pytc
        self.assertTrue(callable(pytc.df.isdf_decompose))
        self.assertTrue(callable(pytc.df.pivoted_cholesky_pair_pivots))


class TestNoCoulombPackageRemains(unittest.TestCase):
    """Gate for task #14's corrective reorganization (isdf-coulomb-cuda,
    2026-07-13): pytc/coulomb/ was consolidated into pytc/integrals/
    coulomb.py with no compatibility shim (Ke's explicit "no Coulomb
    folder" ruling). Alice's acceptance criterion: "an rg gate showing
    no production import still targets pytc.coulomb and no canonical
    implementation remains in root tc/xtc" -- covers import statements
    and module strings in tracked production/test Python sources,
    deliberately excluding git history/build artifacts (this scans the
    actual working tree via `git ls-files`, not disk globbing, so
    untracked scratch files/other agents' worktree checkouts can never
    produce a false failure here).
    """

    def test_pytc_coulomb_package_is_gone(self):
        with self.assertRaises(ModuleNotFoundError):
            import pytc.coulomb  # noqa: F401

    def test_no_pytc_coulomb_references_in_tracked_python_sources(self):
        import os
        import subprocess

        this_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True, cwd=this_dir,
        ).stdout.strip()
        tracked_py_files = subprocess.run(
            ["git", "ls-files", "*.py"],
            capture_output=True, text=True, check=True, cwd=repo_root,
        ).stdout.splitlines()

        offenders = []
        for rel_path in tracked_py_files:
            # This test file itself legitimately names the pattern it
            # is searching for (in this docstring/these string
            # literals) -- exclude it, not a real reference to the old
            # package.
            if rel_path == "pytc/test/test_df_package_compat.py":
                continue
            full_path = f"{repo_root}/{rel_path}"
            with open(full_path, encoding="utf-8") as f:
                for lineno, line in enumerate(f, start=1):
                    if "pytc.coulomb" in line or "pytc/coulomb" in line:
                        offenders.append(f"{rel_path}:{lineno}: {line.strip()}")

        self.assertEqual(
            offenders, [],
            "Found stale pytc.coulomb/pytc/coulomb references in tracked "
            "Python sources after task #14's package deletion:\n"
            + "\n".join(offenders),
        )

    def test_root_tc_xtc_are_shims_not_canonical_implementations(self):
        # The root pytc/tc.py, pytc/xtc.py compatibility shims (task
        # #14 commit A) must stay true sys.modules aliases -- neither
        # should define its own TC/XTC class or grow back into a
        # canonical implementation.
        import pytc.tc
        import pytc.xtc
        import pytc.integrals.tc
        import pytc.integrals.xtc
        self.assertIs(pytc.tc, pytc.integrals.tc)
        self.assertIs(pytc.xtc, pytc.integrals.xtc)

        repo_root_tc = pytc.tc.__file__
        repo_root_xtc = pytc.xtc.__file__
        # Since pytc.tc IS pytc.integrals.tc (same module object), its
        # own __file__ already points at pytc/integrals/tc.py -- confirm
        # that directly rather than re-deriving a path.
        self.assertTrue(repo_root_tc.endswith("pytc/integrals/tc.py"))
        self.assertTrue(repo_root_xtc.endswith("pytc/integrals/xtc.py"))


if __name__ == "__main__":
    unittest.main()
