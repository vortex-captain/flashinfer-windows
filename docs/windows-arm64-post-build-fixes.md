# Windows ARM64 Post-Build Fixes

This document records issues discovered only after building or testing the
first FlashInfer v0.6.14 Windows ARM64 wheel. It complements
`windows-arm64-port-decisions.md`.

## Packaged CCCL SAL identifier transform

**Failure:** The first completed wheel still contained 6,736 exact `__out`
identifiers and zero `__cccl_out` replacements.

**Earlier coverage:** The v0.6.11.post3 repacker patched the extracted wheel
payload directly, so it did not need to account for setuptools source mapping.
This was not an outstanding issue in that workflow.

**Why the initial v0.6.14 port missed it:** The first build hook transformed
the copied `flashinfer/data/cccl` tree, but `pyproject.toml` packages
`flashinfer.data.cccl` directly from `3rdparty/cccl`.

**Fix:** Temporarily replace exact `__out` tokens in the source header while
setuptools assembles the wheel, then restore the original header in a `finally`
block. The transform fails closed unless exactly 6,736 tokens are present.

**Validation:** The corrected wheel contains zero `__out` tokens and 6,736
`__cccl_out` tokens. The CCCL submodule remains clean.

## NVFP4 quantization dispatch NVCC ICE

**Failure:** On the first clean `N=34816` test, the SM121 CUTLASS GEMM module
compiled and loaded, but `fp4_quantization_120f` failed to compile:

```text
quantization.cu(131): Internal Compiler Error (codegen):
"could not lookup variable in map!"
```

**Earlier coverage:** FlashInfer v0.6.11.post3 did not contain
`dispatchNVFP44Over6Config` or the new 4-over-6 specialization matrix in
`quantization.cu`. Its repack changed this file only for `CUtensorMap`
alignment. The earlier Windows NVCC lambda workaround targeted a different
dispatch in `moe_gemm_template_dispatch_tma_ws.h`.

**Why it is new:** FlashInfer v0.6.14 added nested generic-lambda dispatch for
NVFP4 4-over-6 quantization. NVCC 13.4 on Windows ARM64 fails during host
code generation for that new construct.

**Initial fix and follow-up failure:** Replacing only the nested
`dispatchNVFP44Over6Config` lambdas removed the original lookup ICE, but NVCC
then failed later in the same translation unit:

```text
Terminator found in the middle of a basic block
<unnamed>: parse Invalid instruction with no BB
```

The remaining FP4 launch paths still used generic lambdas for layout, row-wise
scale, inverse scale, UE8M0, and kernel-address selection.

**Complete fix:** Replace every generic-lambda dispatch in `quantization.cu`
with named typed adapters and launcher structs. Runtime `if`/`switch` statements
select the same compile-time tags. GPU kernel bodies, launch APIs, launch
geometry, argument order, layouts, optimization flags, and math are unchanged.

**Behavioral coverage:** The rewrite preserves all 18 specializations:

- normal path: two `disableFP4QuantFastMath` values;
- 4-over-6 path: `2 x 2 x 2 x 2` combinations of
  `disableFP4QuantFastMath`, E4M3 maximum (256/448), error mode (MAE/MSE), and
  error fast-math.

**Validation:**

- The isolated SM121 quantization module compiled and linked with CUDA 13.4,
  NVCC LLVM 23, and MSVC ARM64 14.51.
- Source-tree runtime validation completed quantization and the CUTLASS GEMM for
  `M=48, N=34816, K=5120`.
- GEMM time was 7.515 ms and the process exited successfully.
- The final rebuilt wheel installed into `.venv` and passed the same clean-cache
  `N=34816` test. The smoke-test GEMM time was 7.321 ms.
- NumPy-reference validation passed with finite output, cosine similarity
  `0.9909701077`, relative L2 error `0.1343695999`, RMSE `9.612236521`, mean
  absolute error `7.667474716`, and max absolute error `51.10081482`.

## Final wheel

```text
dist/flashinfer_python-0.6.14-py3-none-any.whl
SHA256 42DE5FF238767D0731968C3F85BC3B4FC30BCF6C515D6FBFECC18E232CE9D575
```

Payload checks:

- no native binaries (source-only wheel);
- portable CUTLASS `udiv128` fallback present;
- no indirect SM120 launch wrapper;
- zero CCCL `__out` tokens and 6,736 `__cccl_out` replacements;
- no generic-lambda dispatch remains in `quantization.cu`.
