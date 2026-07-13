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
        # directly by pytc/coulomb/test/test_pivot_selection.py -- an
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
        # pytc/coulomb/ imports shared primitives from the new df/
        # package -- these must keep working unchanged (no relocation
        # of Coulomb's own business logic in this task, per the ratified
        # narrow scope).
        from pytc.coulomb.pivot_selection import select_sector_pivots  # noqa: F401
        from pytc.coulomb.molecular_df_reference import compute_Z  # noqa: F401

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


if __name__ == "__main__":
    unittest.main()
