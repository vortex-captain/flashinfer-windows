"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Shared build utilities for flashinfer packages."""

import subprocess
from pathlib import Path
from typing import Optional


def apply_cutlass_patches(cutlass_dir: Path) -> None:
    """Patch populated CUTLASS sources before packaging or native compilation."""
    header = (
        cutlass_dir
        / "include"
        / "cutlass"
        / "gemm"
        / "kernel"
        / "sm103_blockscaled_gemm_tma_warpspecialized.hpp"
    )
    original = header.read_bytes()
    newline = b"\r\n" if b"\r\n" in original else b"\n"
    before = b"    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);"
    # NoSmem has no epilogue-load producers; only real load pipelines need init.
    after = (
        b"    EpiLoadPipeline epi_load_pipeline(\n"
        b"        shared_storage.pipelines.epi_load,\n"
        b"        epi_load_pipeline_params,\n"
        b"        cute::bool_constant<!IsNoSmemEpilogue>{});"
    ).replace(b"\n", newline)
    if before in original:
        header.write_bytes(original.replace(before, after))
        print(f"Applied CUTLASS SM103 NoSmem pipeline fix: {header}")
    elif after not in original:
        raise RuntimeError(
            f"Expected original or patched SM103 epilogue-load constructor in {header}"
        )


def get_git_version(cwd: Optional[Path] = None) -> str:
    """
    Get git commit hash.

    Args:
        cwd: Working directory for git command. If None, uses current directory.

    Returns:
        Git commit hash or "unknown" if git is not available.
    """
    try:
        git_version = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=cwd,
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
        return git_version
    except Exception:
        return "unknown"


if __name__ == "__main__":
    apply_cutlass_patches(Path(__file__).resolve().parent / "3rdparty" / "cutlass")
