"""Reproduce forward-throughput and numerical-equivalence checks for SWAP fidelity."""

import argparse
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quantum_core import differentiable_entangled_fidelity_vectorized
from src.quantum_fast import fast_entangled_fidelity


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def max_abs_difference(first: torch.Tensor, second: torch.Tensor) -> float:
    return (first - second).abs().max().item()


def measure(
    function: Callable[..., torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
    qubits: int,
    depth: int,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            function(*inputs, qubits, depth)
        synchronize(device)

        timings = []
        for _ in range(repeats):
            synchronize(device)
            started = time.perf_counter()
            function(*inputs, qubits, depth)
            synchronize(device)
            timings.append(time.perf_counter() - started)

    median_seconds = statistics.median(timings)
    return {
        "median_seconds_per_batch": median_seconds,
        "pairs_per_second": inputs[0].shape[0] / median_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--qubits", type=int, default=6)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if min(args.batch_size, args.qubits, args.depth, args.repeats) < 1:
        parser.error("batch-size, qubits, depth, and repeats must be positive")
    if args.warmup < 0:
        parser.error("warmup cannot be negative")

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    generator = torch.Generator(device="cpu").manual_seed(7)
    inputs = (
        (torch.rand(args.batch_size, args.qubits, generator=generator) * 2 - 1).to(device),
        (torch.rand(args.batch_size, args.qubits, generator=generator) * 2 - 1).to(device),
        (torch.randn(args.qubits, generator=generator) * 0.5).to(device),
        (torch.randn(args.qubits, generator=generator) * 0.5).to(device),
    )
    reference = differentiable_entangled_fidelity_vectorized(
        *inputs, args.qubits, args.depth
    )
    optimized = fast_entangled_fidelity(*inputs, args.qubits, args.depth)
    torch.testing.assert_close(optimized, reference, rtol=1e-4, atol=1e-5)
    output_max_abs_error = max_abs_difference(optimized, reference)

    reference_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
    optimized_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
    differentiable_entangled_fidelity_vectorized(
        *reference_inputs, args.qubits, args.depth
    ).sum().backward()
    fast_entangled_fidelity(*optimized_inputs, args.qubits, args.depth).sum().backward()
    gradient_errors = [
        max_abs_difference(fast.grad, reference.grad)
        for fast, reference in zip(optimized_inputs, reference_inputs)
    ]
    for optimized_input, reference_input in zip(optimized_inputs, reference_inputs):
        torch.testing.assert_close(
            optimized_input.grad, reference_input.grad, rtol=1e-3, atol=1e-4
        )

    reference_rate = measure(
        differentiable_entangled_fidelity_vectorized,
        inputs,
        args.qubits,
        args.depth,
        device,
        args.warmup,
        args.repeats,
    )
    optimized_rate = measure(
        fast_entangled_fidelity,
        inputs,
        args.qubits,
        args.depth,
        device,
        args.warmup,
        args.repeats,
    )
    report = {
        "device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "settings": {
            "mode": "forward-only; tensors preallocated on device",
            "batch_size": args.batch_size,
            "qubits": args.qubits,
            "depth": args.depth,
            "warmup_batches": args.warmup,
            "timed_repeats": args.repeats,
            "latency_statistic": "median wall-clock time per batch",
            "synchronize_cuda_before_and_after_each_timed_call": device.type == "cuda",
        },
        "equivalence": {
            "output_atol": 1e-5,
            "output_rtol": 1e-4,
            "output_max_abs_error": output_max_abs_error,
            "gradient_atol": 1e-4,
            "gradient_rtol": 1e-3,
            "gradient_max_abs_error_by_input": gradient_errors,
        },
        "reference": reference_rate,
        "optimized": optimized_rate,
        "speedup": optimized_rate["pairs_per_second"]
        / reference_rate["pairs_per_second"],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
