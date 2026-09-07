# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the FlashInfer project

"""CPU-only coverage of the production FC1 scale-mode forwarding.

Extract the actual function bodies to avoid importing CUDA-dependent package
initializers. Run with --noconftest to avoid the repository's GPU discovery.
Native numerical coverage lives in test_trtllm_cutlass_fused_moe.py.
"""

import __future__
import ast
import functools
import inspect
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


_SOURCE = Path(__file__).resolve().parents[2] / "flashinfer" / "fused_moe" / "core.py"
_OMITTED = object()
_MODES = [_OMITTED, None, False, True]
_MODE_IDS = ["omitted", "none", "shared", "per-expert"]
_FLAG = "fc1_use_per_expert_act_scale"


class _Tensor:
    def __init__(self, shape, dtype="float32", device="cuda:0"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device

    def size(self, dim):
        return self.shape[dim]

    def new_empty(self, shape, dtype):
        return _Tensor(shape, dtype, self.device)

    def __getitem__(self, key):
        raise AssertionError("Dispatch must not index tensor values")

    def __bool__(self):
        raise AssertionError("Dispatch must not inspect tensor values")


def _load_function(path, namespace):
    node = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    for name in path:
        node = next(
            child
            for child in node.body
            if isinstance(child, ast.FunctionDef) and child.name == name
        )
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    exec(
        compile(
            module,
            str(_SOURCE),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace[path[-1]]


def _namespace():
    def empty(shape, *, dtype, device):
        return _Tensor(shape, dtype, device)

    return {
        "functools": functools,
        "ActivationType": SimpleNamespace(Swiglu=0),
        "torch": SimpleNamespace(
            empty=empty,
            int32="int32",
            float32="float32",
            cuda=SimpleNamespace(device=lambda device: nullcontext()),
        ),
        "get_compute_capability": lambda device: (10, 3),
        "device_support_pdl": lambda device: False,
        "check_shape_dtype_device": lambda *args: None,
    }


def _arguments():
    return {
        "output": _Tensor((2, 8)),
        "input": _Tensor((2, 8)),
        "token_selected_experts": _Tensor((2, 2), "int32"),
        "token_final_scales": _Tensor((2, 2)),
        "fc1_expert_weights": _Tensor((4, 16, 8), "int64"),
        "fc1_expert_biases": None,
        "fc2_expert_weights": _Tensor((4, 8, 8), "int64"),
        "fc2_expert_biases": None,
        "output_dtype": "float32",
        "quant_scales": [_Tensor((4,)) for _ in range(6)],
        "input_sf": _Tensor((2, 8)),
        "workspace_buffer": _Tensor((128,), "uint8"),
    }


@pytest.mark.parametrize("mode", _MODES, ids=_MODE_IDS)
def test_public_api_forwards_optional_fc1_mode_without_changing_scales(mode):
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return args[0]

    namespace = _namespace()
    namespace["get_cutlass_fused_moe_module"] = lambda arch: SimpleNamespace(
        cutlass_fused_moe=run
    )
    function = _load_function(("cutlass_fused_moe",), namespace)
    arguments = _arguments()
    if mode is not _OMITTED:
        arguments[_FLAG] = mode

    result = function(**arguments)

    assert inspect.signature(function).parameters[_FLAG].default is None
    assert result is arguments["output"]
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[9] is arguments["quant_scales"]
    assert kwargs[_FLAG] is (None if mode is _OMITTED else mode)
    assert kwargs["workspace_buffer"] is arguments["workspace_buffer"]
    assert arguments["quant_scales"][0].shape == (4,)
    assert arguments["quant_scales"][3].shape == (4,)


@pytest.mark.parametrize("mode", _MODES, ids=_MODE_IDS)
@pytest.mark.parametrize("min_latency_mode", [False, True])
def test_native_dispatch_preserves_legacy_calls_and_passes_explicit_mode(
    mode, min_latency_mode
):
    calls = []
    tuning_calls = []
    runner_arguments = []

    def native_method(name):
        def run(*args):
            calls.append((name, args))

        return run

    legacy_name = "run_moe_min_latency" if min_latency_mode else "run_moe"
    native = SimpleNamespace(**{legacy_name: native_method(legacy_name)})
    explicit = mode is not _OMITTED and mode is not None
    if explicit:
        new_name = legacy_name + "_with_fc1_scale_mode"
        setattr(native, new_name, native_method(new_name))

    class MoERunner:
        tuning_config = object()

        def __init__(self, **kwargs):
            runner_arguments.append(kwargs)
            self.fused_moe_runner = native

        @staticmethod
        def refine_tuning_config(max_tokens):
            pass

    def choose_one(name, runners, config, inputs, **kwargs):
        tuning_calls.append((name, inputs))
        return runners[0], kwargs["gemm_idx"]

    namespace = _namespace()
    namespace["MoERunner"] = MoERunner
    namespace["AutoTuner"] = SimpleNamespace(
        get=lambda: SimpleNamespace(choose_one=choose_one)
    )
    function = _load_function(
        ("get_cutlass_fused_moe_module", "cutlass_fused_moe"), namespace
    )
    arguments = _arguments()
    arguments["min_latency_mode"] = min_latency_mode
    if mode is not _OMITTED:
        arguments[_FLAG] = mode

    result = function(**arguments)

    assert inspect.signature(function).parameters[_FLAG].default is None
    assert len(calls) == 1
    name, args = calls[0]
    if explicit:
        assert name == legacy_name + "_with_fc1_scale_mode"
        assert args[0] is mode
        args = args[1:]
    else:
        # The simulated old module deliberately has no opt-in entry points.
        assert name == legacy_name
    assert len(args) == 26 + (3 if min_latency_mode else 0)
    assert args[8] is arguments["quant_scales"]
    assert args[8][0].shape == (4,)
    assert args[8][3].shape == (4,)
    assert args[-4] == [1, 2]
    assert args[-1] is arguments["workspace_buffer"]
    assert _FLAG not in runner_arguments[0]
    assert [name for name, _ in tuning_calls] == [
        "trtllm::fused_moe::gemm1",
        "trtllm::fused_moe::gemm2",
    ]
    assert all(len(inputs) == 5 for _, inputs in tuning_calls)
    if min_latency_mode:
        assert result is arguments["output"]
    else:
        assert result[0] is arguments["output"]


@pytest.mark.parametrize("mode", [None, False, True])
@pytest.mark.parametrize("min_latency_mode", [False, True])
def test_fake_api_accepts_optional_mode_without_changing_output_shapes(
    mode, min_latency_mode
):
    function = _load_function(
        ("get_cutlass_fused_moe_module", "_fake_cutlass_fused_moe"), _namespace()
    )
    arguments = _arguments()
    arguments[_FLAG] = mode
    arguments["min_latency_mode"] = min_latency_mode

    result = function(**arguments)

    assert inspect.signature(function).parameters[_FLAG].default is None
    assert result[0].shape == ((8, 8) if min_latency_mode else (2, 8))
    assert arguments["quant_scales"][0].shape == (4,)
