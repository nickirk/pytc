"""Durability and coverage gates for physical-q shard artifacts."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pytc.pbc.df.q_shards import (
    assemble_q_shard_artifact_set,
    read_q_shard_artifact,
    validate_q_shard_artifact_set,
    write_q_shard_artifact,
)


def _shard(q_indices, *, n_kpts=4, n_ip=3, source="source-a"):
    q_indices = np.asarray(q_indices, dtype=np.int64)
    base = np.empty((q_indices.size, n_ip, n_ip), dtype=np.complex128)
    for row, q in enumerate(q_indices):
        base[row] = (q + 1) * np.eye(n_ip) + 1j * (q + 2)
    return {
        "q_indices": q_indices,
        "coul_kpt": base,
        "kern_kpt": base + (10.0 + 2.0j),
        "solve_infos": [{"q": int(q), "n_retained": n_ip} for q in q_indices],
        "kernel_provider": {"name": "synthetic"},
        "shard_provenance": {
            "schema_version": 1,
            "source_fingerprint": source,
            "mesh_fingerprint": "mesh-a",
            "n_kpts": n_kpts,
            "q_indices": q_indices.tolist(),
            "q_block_size": int(q_indices.size),
            "block_size": 17,
            "p_block_rows": 2,
            "retention_mode": "single",
            "rtol": None,
            "n_retained_pin": n_ip,
            "jitter_rcond": None,
            "transform": "mapped_fft_selected_q_v1",
        },
    }


class TestQShardArtifacts(unittest.TestCase):
    def test_round_trip_is_mmap_and_complete_set_assembles_in_q_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard_20 = _shard([2, 0])
            shard_31 = _shard([3, 1])
            write_q_shard_artifact(root / "shard-20", shard_20)
            write_q_shard_artifact(root / "shard-31", shard_31)

            manifest, arrays = read_q_shard_artifact(root / "shard-20")
            self.assertEqual(manifest["q_indices"], [2, 0])
            self.assertIsInstance(arrays["coul_kpt"], np.memmap)
            np.testing.assert_array_equal(arrays["coul_kpt"], shard_20["coul_kpt"])

            assembled = assemble_q_shard_artifact_set(
                [root / "shard-31", root / "shard-20"],
                root / "assembled",
                expected_n_kpts=4,
                source_fingerprint="source-a",
            )
            self.assertTrue(assembled["completed"])
            coul = np.load(root / "assembled" / "coul_kpt.npy", mmap_mode="r")
            kern = np.load(root / "assembled" / "kern_kpt.npy", mmap_mode="r")
            expected_coul = np.empty((4, 3, 3), dtype=np.complex128)
            expected_kern = np.empty_like(expected_coul)
            for shard in (shard_20, shard_31):
                expected_coul[shard["q_indices"]] = shard["coul_kpt"]
                expected_kern[shard["q_indices"]] = shard["kern_kpt"]
            np.testing.assert_array_equal(coul, expected_coul)
            np.testing.assert_array_equal(kern, expected_kern)

    def test_duplicate_missing_and_mixed_source_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_q_shard_artifact(root / "a", _shard([0, 1]))
            write_q_shard_artifact(root / "duplicate", _shard([1, 2]))
            write_q_shard_artifact(root / "other-source", _shard([2, 3], source="source-b"))

            with self.assertRaisesRegex(ValueError, "appears in both"):
                validate_q_shard_artifact_set(
                    [root / "a", root / "duplicate"], expected_n_kpts=4
                )
            with self.assertRaisesRegex(ValueError, "incomplete"):
                validate_q_shard_artifact_set([root / "a"], expected_n_kpts=4)
            with self.assertRaisesRegex(ValueError, "construction contract"):
                validate_q_shard_artifact_set(
                    [root / "a", root / "other-source"], expected_n_kpts=4
                )

    def test_corruption_incomplete_directory_and_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = root / "shard"
            write_q_shard_artifact(artifact, _shard([0, 1]))
            with self.assertRaises(FileExistsError):
                write_q_shard_artifact(artifact, _shard([0, 1]))

            with open(artifact / "coul_kpt.npy", "ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "byte count mismatch"):
                read_q_shard_artifact(artifact)

            incomplete = root / "incomplete"
            incomplete.mkdir()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                read_q_shard_artifact(incomplete)

    def test_manifest_payload_cannot_be_relabelled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "shard"
            write_q_shard_artifact(artifact, _shard([0, 1]))
            manifest_path = artifact / "manifest.json"
            with open(manifest_path, "rb") as stream:
                manifest = json.load(stream)
            manifest["q_indices"] = [2, 3]
            with open(manifest_path, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream)
            with self.assertRaisesRegex(ValueError, "disagree"):
                read_q_shard_artifact(artifact)


if __name__ == "__main__":
    unittest.main()
