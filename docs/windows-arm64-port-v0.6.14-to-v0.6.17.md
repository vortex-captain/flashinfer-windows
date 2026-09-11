# Windows ARM64 Port: FlashInfer v0.6.14 to v0.6.17

This document records automatic conflict resolutions and new compatibility
fixes while replaying `win-arm64-v0.6.14` onto official FlashInfer v0.6.17.
Agent-authored adaptations must preserve v0.6.17 semantics, behavior, and
performance.

## Baseline

- Official v0.6.17 commit:
  `a0a6b019b9b27d49d209f85d028a1ae5a9b347d7`
- Target branch: `win-arm64-v0.6.17`
- Replayed source branch: `win-arm64-v0.6.14`
- CUTLASS pin retained:
  `b46b16d003484063bca4ed365e44095c4c6ed633`

## `338ce53d` conflict resolutions

### `.gitignore`

**Resolution:** Keep v0.6.17's newer split-EP build paths and add only the
Windows `*/compiled_cache.db` pattern.

**Potential impact:** None at runtime. Avoids reintroducing stale pre-v0.6.17
directory patterns.

### `flashinfer/jit/xqa.py`

**Resolution:** Keep v0.6.17's descriptive XQA JIT URI, including its
ragged-Q suffix behavior.

**Potential impact:** Preserves v0.6.17 cache identity and speculative/ragged-Q
specialization. The shorter Windows-fork URI was not required for correctness
and would merge distinct v0.6.17 builds.

### `flashinfer/xqa.py`

**Resolution:** Keep v0.6.17's `op_name = f"flashinfer::{spec.name}"` for both
the real and fake operators.

**Potential impact:** Preserves the invariant that JIT module and Torch
operator names cannot drift. Avoids the older branch's undefined
`use_spec_dec` reference.

### `requirements.txt`

**Resolution:** Keep v0.6.17's `nvidia-cudnn-frontend>=1.25.0` floor and add
the existing Windows exclusion marker only to `nvidia-cutlass-dsl>=4.5.0`.

**Potential impact:** Non-Windows dependency behavior is unchanged. Native
Windows avoids resolving unavailable CUTLASS DSL packages; CUTLASS C++ JIT
backends remain available.

## `579f9995` conflict resolutions

### TRTLLM custom-routing shared memory

**Resolution:** Keep v0.6.17's caller-provided
`smemPackedScoreIdx` parameter. Do not reintroduce the older local shared-memory
array that would shadow the parameter.

**Potential impact:** Preserves v0.6.17 cluster-kernel storage ownership and
layout.

### TRTLLM BMM Windows header overlays

**Resolution:** Combine the Windows LLP64 header overlay with v0.6.17's
per-module Blackwell/Rubin export isolation. Each Windows overlay is now keyed
by `module_name`; non-Windows retains the per-module symlink root.

**Potential impact:** Header content and kernel selection are unchanged. This
prevents Blackwell and Rubin generation from sharing or overwriting an export
header tree during AOT builds.

## Cleanly replayed commits

The following v0.6.14 port commits applied to v0.6.17 without conflict:

- `6ad8f76e` — C++ standard flag ordering
- `37dcaccc` — quoted Windows Ninja/CUDA paths
- `689f59b8` — short Windows JIT cache path
- `9795486e` — temporary CCCL wheel transform, later superseded by the shared
  runtime-JIT/JIT-cache generated include overlay
- `e57cad47` — initial NVFP4 dispatch ICE workaround
- `23556d73` — complete quantization dispatch refactor
- `ab8049b1` — historical v0.6.14 validation record

## v0.6.17 source checks

- CUTLASS remains pinned to `b46b16d` with portable `uint128` division.
- The unproven indirect SM120/SM121 launch wrapper remains excluded.
- No generic-lambda dispatch remains in `quantization.cu`.
- Changed Python build/JIT modules pass `py_compile`.
- The v0.6.17 descriptive XQA URI and shared real/fake operator name remain
  intact.

## Pending validation

- Wheel build and source-payload checks
- SM121 `N=34816` NumPy correctness validation
