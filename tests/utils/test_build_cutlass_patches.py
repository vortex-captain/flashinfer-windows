# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the FlashInfer project

"""CPU-only build coverage; use --noconftest to avoid GPU discovery."""

import ast
import shutil
from pathlib import Path

import pytest

import build_utils


_ROOT = Path(__file__).resolve().parents[2]
_MAIN_BACKEND = _ROOT / "build_backend.py"
_HEADER = (
    Path("include")
    / "cutlass"
    / "gemm"
    / "kernel"
    / "sm103_blockscaled_gemm_tma_warpspecialized.hpp"
)
_BEFORE = b"    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);"
_AFTER = (
    b"    EpiLoadPipeline epi_load_pipeline(\n"
    b"        shared_storage.pipelines.epi_load,\n"
    b"        epi_load_pipeline_params,\n"
    b"        cute::bool_constant<!IsNoSmemEpilogue>{});"
)
_PREFIX = b"    epi_load_pipeline_params.initializing_warp = 4;\n"
_SUFFIX = b"\n\n    // Epilogue Store pipeline\n"


def _write_header(root, content):
    header = root / _HEADER
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_bytes(content)
    return header


def _load_functions(source, names, namespace):
    # Execute production functions without the backends' import-time metadata
    # writes or CUDA-dependent imports.
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(nodes) == len(names)
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace
    )


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
@pytest.mark.parametrize("already_patched", [False, True])
def test_exact_patch_and_idempotence(tmp_path, newline, already_patched):
    expected = (_PREFIX + _AFTER + _SUFFIX).replace(b"\n", newline)
    original = (
        expected
        if already_patched
        else (_PREFIX + _BEFORE + _SUFFIX).replace(b"\n", newline)
    )
    header = _write_header(tmp_path, original)
    original_mtime = header.stat().st_mtime_ns

    build_utils.apply_cutlass_patches(tmp_path)
    assert header.read_bytes() == expected
    if already_patched:
        assert header.stat().st_mtime_ns == original_mtime
    patched_mtime = header.stat().st_mtime_ns
    build_utils.apply_cutlass_patches(tmp_path)
    assert header.stat().st_mtime_ns == patched_mtime
    assert header.read_bytes() == expected


def test_missing_header_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_utils.apply_cutlass_patches(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_source_drift_does_not_modify_header(tmp_path):
    content = _BEFORE.replace(b"epi_load_pipeline_params", b"changed_params")
    header = _write_header(tmp_path, content)
    with pytest.raises(RuntimeError, match="Expected original or patched"):
        build_utils.apply_cutlass_patches(tmp_path)
    assert header.read_bytes() == content
    assert list(header.parent.iterdir()) == [header]


@pytest.mark.parametrize("kind", ["wheel", "editable", "sdist", "aot"])
def test_shared_checkout_is_patched_before_copy_or_link(tmp_path, monkeypatch, kind):
    cutlass = tmp_path / "3rdparty" / "cutlass"
    header = _write_header(cutlass, _BEFORE)
    data = tmp_path / "flashinfer" / "data"
    links = []

    def symlink_to(destination, source, *, target_is_directory):
        assert header.read_bytes() == _AFTER
        assert target_is_directory
        links.append((destination, source))

    monkeypatch.setattr(Path, "symlink_to", symlink_to)
    namespace = {
        "_root": tmp_path,
        "_data_dir": data,
        "shutil": shutil,
        "apply_cutlass_patches": build_utils.apply_cutlass_patches,
        "_install_cuda_tile_compile_deps": lambda: None,
        "_build_nvep_if_enabled": lambda: None,
    }
    names = ["_create_data_dir"] + ([] if kind == "aot" else [f"_prepare_for_{kind}"])
    _load_functions(_MAIN_BACKEND, names, namespace)
    for _ in range(2):
        if kind == "aot":
            # The AOT backend calls this helper after populating submodules.
            namespace["_create_data_dir"](use_symlinks=True)
        else:
            namespace[f"_prepare_for_{kind}"]()
        assert header.read_bytes() == _AFTER
        if kind in ("editable", "aot"):
            assert (data / "cutlass", cutlass) in links
        else:
            assert (data / "cutlass" / _HEADER).read_bytes() == _AFTER
