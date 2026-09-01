"""Durable, streamable artifacts for bounded physical-q ISDF shards.

The large-mesh contract is deliberately directory based: each array is a
standalone ``.npy`` file that can be memory-mapped, and ``manifest.json`` is
written last.  A missing manifest therefore means an incomplete shard, while a
complete shard can be validated and consumed without gathering all q rows.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np


_SCHEMA_VERSION = 1
_ARRAY_NAMES = ("coul_kpt", "kern_kpt")


def _jsonable(value, *, path="value"):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return _jsonable(value.item(), path=path)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist(), path=path)
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} has non-string key {key!r}.")
            out[key] = _jsonable(item, path=f"{path}.{key}")
        return out
    if isinstance(value, (list, tuple)):
        return [
            _jsonable(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} contains non-JSON value {type(value).__name__}.")


def _canonical_json(value):
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_file(path, *, chunk_bytes=8 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _update_array_digest(digest, label, value):
    array = np.ascontiguousarray(value)
    digest.update(label.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(array).cast("B"))


def fingerprint_kpts_mesh(mesh_obj):
    """Fingerprint every mesh field that fixes physical-q interpretation."""
    digest = hashlib.sha256()
    digest.update(b"pytc-kpts-mesh-v1")
    digest.update(np.asarray(mesh_obj.kmesh, dtype=np.int64).tobytes())
    for name in (
        "canonical_kpts",
        "neg",
        "fft_k_indices",
        "fft_r_indices",
    ):
        value = getattr(mesh_obj, name, None)
        if value is None:
            raise ValueError(
                f"mesh_obj.{name} is required for a durable q-shard artifact."
            )
        _update_array_digest(digest, name, value)
    return digest.hexdigest()


def fingerprint_inpv(inpv_kpt):
    """Fingerprint the logical complex128 interpolation-value tensor."""
    array = np.asarray(inpv_kpt)
    if array.ndim != 3 or array.dtype != np.complex128 or not array.size:
        raise ValueError("inpv_kpt must be a nonempty complex128 rank-3 array.")
    digest = hashlib.sha256()
    digest.update(b"pytc-inpv-kpt-v1")
    _update_array_digest(digest, "inpv_kpt", array)
    return digest.hexdigest()


def _validated_shard_payload(shard):
    if not isinstance(shard, dict):
        raise TypeError("shard must be the dict returned by build_q_shard_from_inpv.")
    try:
        q_indices = np.asarray(shard["q_indices"])
        provenance = _jsonable(
            shard["shard_provenance"], path="shard.shard_provenance"
        )
        kernel_provider = _jsonable(
            shard["kernel_provider"], path="shard.kernel_provider"
        )
        solve_infos = _jsonable(shard["solve_infos"], path="shard.solve_infos")
    except KeyError as exc:
        raise ValueError(f"shard is missing required field {exc.args[0]!r}.") from exc

    if q_indices.ndim != 1 or not np.issubdtype(q_indices.dtype, np.integer):
        raise ValueError("shard q_indices must be a 1-D integer array.")
    q_indices = q_indices.astype(np.int64, copy=False)
    if q_indices.size == 0 or np.unique(q_indices).size != q_indices.size:
        raise ValueError("shard q_indices must be nonempty and unique.")
    n_kpts = int(provenance.get("n_kpts", -1))
    if np.any(q_indices < 0) or np.any(q_indices >= n_kpts):
        raise ValueError(f"shard q_indices must lie in [0,{n_kpts}).")
    if provenance.get("q_indices") != q_indices.tolist():
        raise ValueError("shard q_indices disagree with shard_provenance.")
    if len(solve_infos) != q_indices.size:
        raise ValueError("solve_infos length must equal the q-shard size.")

    arrays = {}
    n_ip = None
    for name in _ARRAY_NAMES:
        try:
            array = np.asarray(shard[name])
        except KeyError as exc:
            raise ValueError(f"shard is missing required field {name!r}.") from exc
        if array.ndim != 3 or array.shape[0] != q_indices.size:
            raise ValueError(
                f"{name} must have shape (Bq,Nip,Nip) with Bq={q_indices.size}."
            )
        if array.shape[1] != array.shape[2]:
            raise ValueError(f"{name} must be square on its last two axes.")
        if array.dtype != np.complex128:
            raise ValueError(f"{name} must be complex128, got {array.dtype}.")
        if n_ip is None:
            n_ip = int(array.shape[1])
        elif array.shape[1] != n_ip:
            raise ValueError("coul_kpt and kern_kpt must share Nip.")
        arrays[name] = array

    source_fingerprint = provenance.get("source_fingerprint")
    mesh_fingerprint = provenance.get("mesh_fingerprint")
    for name, value in (
        ("source_fingerprint", source_fingerprint),
        ("mesh_fingerprint", mesh_fingerprint),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"shard_provenance.{name} must be a nonempty string.")

    contract_provenance = dict(provenance)
    contract_provenance.pop("q_indices", None)
    contract_provenance.pop("q_block_size", None)
    contract = {
        "schema_version": _SCHEMA_VERSION,
        "source_fingerprint": source_fingerprint,
        "mesh_fingerprint": mesh_fingerprint,
        "n_kpts": n_kpts,
        "n_ip": n_ip,
        "dtype": "complex128",
        "kernel_provider": kernel_provider,
        "shard_contract": contract_provenance,
    }
    contract_sha256 = hashlib.sha256(_canonical_json(contract)).hexdigest()
    return q_indices, arrays, provenance, kernel_provider, solve_infos, contract, contract_sha256


def write_q_shard_artifact(directory, shard):
    """Write one immutable q-shard directory and return its manifest.

    ``directory`` must not exist.  Array files are flushed and checksummed
    before the manifest is atomically installed as the terminal receipt.
    """
    root = Path(directory)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite q-shard artifact {root}.")
    if not root.parent.is_dir():
        raise FileNotFoundError(f"q-shard parent directory does not exist: {root.parent}.")

    (
        q_indices,
        arrays,
        provenance,
        kernel_provider,
        solve_infos,
        contract,
        contract_sha256,
    ) = _validated_shard_payload(shard)
    os.mkdir(root)

    array_records = {}
    for name, array in arrays.items():
        filename = f"{name}.npy"
        path = root / filename
        with open(path, "wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        array_records[name] = {
            "filename": filename,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }

    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "completed": True,
        "contract": contract,
        "contract_sha256": contract_sha256,
        "q_indices": q_indices.tolist(),
        "arrays": array_records,
        "solve_infos": solve_infos,
        "kernel_provider": kernel_provider,
        "shard_provenance": provenance,
    }
    manifest_tmp = root / "manifest.json.tmp"
    with open(manifest_tmp, "wb") as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
        stream.write(b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(manifest_tmp, root / "manifest.json")
    return manifest


def read_q_shard_artifact(directory, *, verify_files=True, mmap_mode="r"):
    """Validate and open one completed shard without materialising its arrays."""
    root = Path(directory)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"q-shard artifact is incomplete: missing {manifest_path}.")
    with open(manifest_path, "rb") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(
            f"unsupported q-shard schema_version={manifest.get('schema_version')!r}."
        )
    if manifest.get("completed") is not True:
        raise ValueError("q-shard manifest is not terminal (completed != true).")
    contract = manifest.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("q-shard manifest contract must be a JSON object.")
    expected_contract_sha = hashlib.sha256(_canonical_json(contract)).hexdigest()
    if manifest.get("contract_sha256") != expected_contract_sha:
        raise ValueError("q-shard contract_sha256 does not match the manifest contract.")

    q_indices = np.asarray(manifest.get("q_indices"))
    if q_indices.ndim != 1 or not np.issubdtype(q_indices.dtype, np.integer):
        raise ValueError("manifest q_indices must be a 1-D integer array.")
    q_indices = q_indices.astype(np.int64, copy=False)
    if q_indices.size == 0 or np.unique(q_indices).size != q_indices.size:
        raise ValueError("manifest q_indices must be nonempty and unique.")

    arrays = {}
    records = manifest.get("arrays")
    if not isinstance(records, dict) or set(records) != set(_ARRAY_NAMES):
        raise ValueError(f"q-shard arrays must be exactly {_ARRAY_NAMES}.")
    for name in _ARRAY_NAMES:
        record = records[name]
        filename = record.get("filename")
        if filename != f"{name}.npy" or Path(filename).name != filename:
            raise ValueError(f"unsafe q-shard filename for {name}: {filename!r}.")
        path = root / filename
        if not path.is_file():
            raise ValueError(f"q-shard array is missing: {path}.")
        if int(path.stat().st_size) != int(record.get("bytes", -1)):
            raise ValueError(f"q-shard byte count mismatch for {name}.")
        if verify_files and _sha256_file(path) != record.get("sha256"):
            raise ValueError(f"q-shard SHA-256 mismatch for {name}.")
        array = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
        if list(array.shape) != record.get("shape") or str(array.dtype) != record.get(
            "dtype"
        ):
            raise ValueError(f"q-shard array header mismatch for {name}.")
        arrays[name] = array

    shard = {
        "q_indices": q_indices,
        "coul_kpt": arrays["coul_kpt"],
        "kern_kpt": arrays["kern_kpt"],
        "solve_infos": manifest.get("solve_infos"),
        "kernel_provider": manifest.get("kernel_provider"),
        "shard_provenance": manifest.get("shard_provenance"),
    }
    try:
        (*_, rebuilt_contract, rebuilt_contract_sha) = _validated_shard_payload(shard)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid q-shard manifest payload: {exc}") from exc
    if contract != rebuilt_contract or expected_contract_sha != rebuilt_contract_sha:
        raise ValueError("q-shard manifest contract disagrees with its payload.")
    return manifest, arrays


def validate_q_shard_artifact_set(
    directories,
    *,
    expected_n_kpts,
    source_fingerprint=None,
    verify_files=True,
    require_complete=True,
):
    """Validate disjoint shard coverage and return artifacts in q-min order."""
    if isinstance(expected_n_kpts, bool) or not isinstance(
        expected_n_kpts, (int, np.integer)
    ):
        raise ValueError("expected_n_kpts must be a positive integer.")
    expected_n_kpts = int(expected_n_kpts)
    if expected_n_kpts <= 0:
        raise ValueError("expected_n_kpts must be a positive integer.")
    directories = [Path(path) for path in directories]
    if not directories:
        raise ValueError("at least one q-shard artifact is required.")
    opened = [
        (path, *read_q_shard_artifact(path, verify_files=verify_files))
        for path in directories
    ]
    contract_sha = opened[0][1]["contract_sha256"]
    source = opened[0][1]["contract"]["source_fingerprint"]
    if source_fingerprint is not None and source != source_fingerprint:
        raise ValueError("q-shard source_fingerprint does not match the requested source.")

    owner = {}
    for path, manifest, _ in opened:
        contract = manifest["contract"]
        if manifest["contract_sha256"] != contract_sha:
            raise ValueError("q-shard artifacts do not share one construction contract.")
        if int(contract["n_kpts"]) != expected_n_kpts:
            raise ValueError("q-shard n_kpts does not match expected_n_kpts.")
        for q in manifest["q_indices"]:
            q = int(q)
            if q in owner:
                raise ValueError(
                    f"physical q={q} appears in both {owner[q]} and {path}."
                )
            owner[q] = path

    expected = set(range(expected_n_kpts))
    observed = set(owner)
    if not observed.issubset(expected):
        raise ValueError("q-shard set contains out-of-range physical q indices.")
    if require_complete and observed != expected:
        missing = sorted(expected - observed)
        raise ValueError(f"q-shard set is incomplete; missing physical q rows {missing}.")
    return sorted(opened, key=lambda item: min(item[1]["q_indices"]))


def assemble_q_shard_artifact_set(
    directories,
    output_directory,
    *,
    expected_n_kpts,
    source_fingerprint=None,
    verify_files=True,
):
    """Assemble a complete shard set into two disk-backed canonical-q arrays.

    The output directory must not exist.  Array files are populated through
    ``open_memmap`` and ``manifest.json`` is installed last, so an interrupted
    assembly is visibly incomplete and is never mistaken for a terminal one.
    """
    opened = validate_q_shard_artifact_set(
        directories,
        expected_n_kpts=expected_n_kpts,
        source_fingerprint=source_fingerprint,
        verify_files=verify_files,
        require_complete=True,
    )
    root = Path(output_directory)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite q-shard assembly {root}.")
    if not root.parent.is_dir():
        raise FileNotFoundError(
            f"q-shard assembly parent directory does not exist: {root.parent}."
        )

    first_manifest = opened[0][1]
    contract = first_manifest["contract"]
    n_kpts = int(contract["n_kpts"])
    n_ip = int(contract["n_ip"])
    os.mkdir(root)

    destinations = {}
    for name in _ARRAY_NAMES:
        destinations[name] = np.lib.format.open_memmap(
            root / f"{name}.npy",
            mode="w+",
            dtype=np.complex128,
            shape=(n_kpts, n_ip, n_ip),
        )
    for _, manifest, arrays in opened:
        q_indices = np.asarray(manifest["q_indices"], dtype=np.int64)
        for name in _ARRAY_NAMES:
            destinations[name][q_indices] = arrays[name]
    for array in destinations.values():
        array.flush()
    del destinations

    array_records = {}
    for name in _ARRAY_NAMES:
        path = root / f"{name}.npy"
        array_records[name] = {
            "filename": path.name,
            "shape": [n_kpts, n_ip, n_ip],
            "dtype": "complex128",
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }
    shard_records = []
    for path, manifest, _ in opened:
        manifest_path = path / "manifest.json"
        shard_records.append(
            {
                "directory": str(path),
                "q_indices": manifest["q_indices"],
                "manifest_sha256": _sha256_file(manifest_path),
            }
        )
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "completed": True,
        "kind": "canonical_q_assembly",
        "contract": contract,
        "contract_sha256": first_manifest["contract_sha256"],
        "arrays": array_records,
        "shards": shard_records,
    }
    manifest_tmp = root / "manifest.json.tmp"
    with open(manifest_tmp, "wb") as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
        stream.write(b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(manifest_tmp, root / "manifest.json")
    return manifest
