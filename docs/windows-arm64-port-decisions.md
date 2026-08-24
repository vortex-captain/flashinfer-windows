# Windows ARM64 Port Decisions

This document records conflict resolutions made while porting the
`flashinfer-windows` changes from `v0.6.11.post3` onto FlashInfer `v0.6.14`.
It is intended to help diagnose later build or runtime regressions.

## Port scope

- Base: official FlashInfer `v0.6.14` (`19f1a41e6b21f0c422d775e377b6fdf9a1fc9d23`)
- Windows range: `3ea119e2f927927a89e5c7df9bb9bb7d18673dd3` through
  `713358284345314df4f40ddc352f4e981f5bb03e`, inclusive
- Repack inputs: `flashinfer-win-arm64-repack-20260824-142131.zip`
- Primary validation: Windows ARM64 SM121 NVFP4 GEMM with `N=34816`

`3ea119e` is a merge commit. It is replayed with mainline parent 2 because
parent 2 is the upstream FlashInfer v0.6.11.post3 side; this applies the merge
result relative to upstream and avoids replaying the unrelated v0.6.8-to-v0.6.11
upstream evolution.

## Conflict decisions

### `.gitignore`

**Decision:** Keep both the v0.6.14 NVEP build-artifact patterns and the Windows
`*/compiled_cache.db` pattern.

**Potential impact:** None at runtime. This only prevents generated local files
from being committed.

### `.gitmodules`

**Decision:** Keep all v0.6.14 submodules, including NIXL and NCCL. The CCCL URL
was identical on both sides.

**Potential impact:** Preserves v0.6.14 NVEP functionality. Removing NIXL/NCCL
would make the source tree inconsistent with its build backend.

### `csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh`

**Decision:** Port the Windows function-pointer dispatch workaround while
preserving every v0.6.14 activation, fast-math, and NVFP4 4-over-6 variant.
Sweep new v0.6.14 dispatch code for the same NVCC pattern and apply only
behavior-neutral rewrites.

**Potential impact:** Intended to avoid Windows ARM64 NVCC template/lambda code
generation failures without changing selected kernels. Incorrect template
coverage could appear as a missing tactic, compile failure, or activation
mismatch, so all affected translation units must be compiled.

### `flashinfer/gdn_kernels/__init__.py`

**Decision:** Keep the guarded v0.6.14 `delta_rule_dsl` imports. Missing CUTLASS
DSL raises `ImportError` or `RuntimeError`, after which the exported symbols are
set to `None`.

**Potential impact:** None for `FlashInferCutlassNvFp4LinearKernel`; GDN is a
separate subsystem. Keeping the guards avoids undefined names in `__all__`.

### `version.txt`

**Decision:** Keep `0.6.14`.

**Potential impact:** Preserves correct API and wheel provenance. A Windows
local-version suffix may still be supplied at build time without changing the
source version.

### SM120/SM121 indirect CUTLASS launch

**Decision:** Do not port the repack's Windows ARM64-only indirect
`Params const*` kernel wrapper. Keep the direct CUTLASS launch from FlashInfer
v0.6.14.

**Reason:** Controlled direct-versus-indirect testing produced the same matrix:
both launched `N=65536` successfully and both terminated at `N=34816` with
`0xc0000409`. The minidump located that failure in CUTLASS host-side
`uint128_t` division before kernel launch. FlashInfer v0.6.14 pins a newer
CUTLASS with the portable division fallback.

**Potential impact:** This removes an extra host-to-device parameter copy and
wrapper kernel without losing a demonstrated benefit. If direct launch later
exposes a distinct Windows ARM64 over-aligned-parameter ABI failure after
argument construction succeeds, the indirect wrapper may need to be revisited
as a separate targeted fix.

### CUTLASS submodule pin

**Decision:** Keep FlashInfer v0.6.14's NVIDIA/CUTLASS pin
`b46b16d003484063bca4ed365e44095c4c6ed633` (CUTLASS 4.5-era source). Do not
take the `3ea119e` downgrade to
`e6e2cc29f5e7611dfc6af0ed6409209df0068cf2` (CUTLASS 4.2.1).

**Reason:** CUTLASS 4.2.1 lacks the portable `uint128` multiply/divide fallback
and aborts on Windows ARM64 for non-power-of-two `FastDivmodU64` divisors.
CUTLASS 4.5 contains that arithmetic fix.

**Potential impact:** CUTLASS 4.5 is provisional, not proven comprehensively on
Windows ARM64. Other compiler, alignment, ABI, scheduler, or kernel regressions
may require exhaustive trial-and-error across later or intermediate CUTLASS
commits. Preserve the `uint128` fix when testing alternate pins, and validate
all affected JIT modules rather than treating the successful NVFP4 probe as
general compatibility proof.

## Pending decisions

- Review of other cleanly applied `3ea119e` changes, especially older GDN
  additions
- Any conflicts from `33d131a`, `502d849`, and `7133582`
- Repack patch and scripted transformation conflicts
