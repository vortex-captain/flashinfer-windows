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

## Pending

- Remaining replay commits
- Wheel build and source-payload checks
- SM121 `N=34816` NumPy correctness validation
- vLLM NVFP4 end-to-end validation with 25GB KV cache
