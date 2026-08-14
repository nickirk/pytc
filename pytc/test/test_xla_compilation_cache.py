"""Focused safety tests for the opt-in XLA persistent cache."""

import os
import sys
import tempfile
import unittest
from unittest import mock

from pytc.utils import gpu_memory


class _FakeJax:
    def __init__(self, has_gpu=True):
        self.devices = mock.Mock(return_value=[object()] if has_gpu else [])
        self.config = mock.Mock()


class TestXlaCompilationCache(unittest.TestCase):
    def setUp(self):
        self._prior_enabled = gpu_memory._XLA_CACHE_ENABLED
        gpu_memory._XLA_CACHE_ENABLED = False
        self.addCleanup(setattr, gpu_memory, "_XLA_CACHE_ENABLED",
                        self._prior_enabled)
        self.addCleanup(os.environ.pop, gpu_memory._XLA_CACHE_DIR_ENV, None)

    def _enable_with_fake_jax(self, *args, **kwargs):
        fake_jax = _FakeJax()
        with mock.patch.dict(sys.modules, {"jax": fake_jax}):
            gpu_memory.enable_xla_compilation_cache(*args, **kwargs)
        return fake_jax

    def test_unset_cache_dir_never_queries_home_or_configures_jax(self):
        fake_jax = _FakeJax()
        with (
            mock.patch.dict(sys.modules, {"jax": fake_jax}),
            mock.patch.object(gpu_memory.os.path, "expanduser",
                              side_effect=AssertionError("home lookup")),
        ):
            gpu_memory.enable_xla_compilation_cache()

        fake_jax.config.update.assert_not_called()
        self.assertFalse(gpu_memory._XLA_CACHE_ENABLED)

    def test_environment_cache_dir_is_created_and_configured(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = os.path.join(temp_dir, "jax-cache")
            os.environ[gpu_memory._XLA_CACHE_DIR_ENV] = cache_dir
            fake_jax = self._enable_with_fake_jax()

            self.assertTrue(os.path.isdir(cache_dir))
            self.assertEqual(
                fake_jax.config.update.call_args_list[0].args,
                ("jax_compilation_cache_dir", cache_dir),
            )
            self.assertTrue(gpu_memory._XLA_CACHE_ENABLED)

    def test_relative_cache_dir_is_rejected(self):
        os.environ[gpu_memory._XLA_CACHE_DIR_ENV] = "relative-cache"

        with self.assertRaisesRegex(ValueError, "absolute path"):
            self._enable_with_fake_jax()

        self.assertFalse(gpu_memory._XLA_CACHE_ENABLED)

    def test_explicit_cache_dir_overrides_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_dir = os.path.join(temp_dir, "from-env")
            explicit_dir = os.path.join(temp_dir, "explicit")
            os.environ[gpu_memory._XLA_CACHE_DIR_ENV] = env_dir
            fake_jax = self._enable_with_fake_jax(explicit_dir)

            self.assertTrue(os.path.isdir(explicit_dir))
            self.assertFalse(os.path.exists(env_dir))
            self.assertEqual(
                fake_jax.config.update.call_args_list[0].args,
                ("jax_compilation_cache_dir", explicit_dir),
            )


if __name__ == "__main__":
    unittest.main()
