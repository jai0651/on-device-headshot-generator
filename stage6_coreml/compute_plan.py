"""Stage 6c: which device does every operation of a Core ML model actually run on?

    .venv-coreml/bin/python -m stage6_coreml.compute_plan --mlpackage <path> [--units CPU_AND_NE]

Section 05: on the Neural Engine, unsupported ops don't fail, they fall back to GPU or CPU
mid-graph, and the hand-off costs more than the op. Contiguity matters more than percentage.
This is the scripted equivalent of Xcode's performance report: per op, the preferred device
Core ML chose, then totals by device and by op type, plus how many times execution switches
device while walking the graph in order.
"""

import argparse
import json
from collections import Counter

import coremltools as ct
from coremltools.models.compute_device import MLCPUComputeDevice, MLGPUComputeDevice, MLNeuralEngineComputeDevice
from coremltools.models.compute_plan import MLComputePlan

UNITS = {"CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE, "ALL": ct.ComputeUnit.ALL, "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU}


def device_name(d):
    return {MLNeuralEngineComputeDevice: "ANE", MLGPUComputeDevice: "GPU", MLCPUComputeDevice: "CPU"}.get(type(d), "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlpackage", required=True)
    ap.add_argument("--units", default="CPU_AND_NE")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # keep `model` alive: its compiled .mlmodelc is a temp dir deleted when the object is collected
    model = ct.models.MLModel(args.mlpackage, compute_units=UNITS[args.units])
    compiled = model.get_compiled_model_path()
    plan = MLComputePlan.load_from_path(path=compiled, compute_units=UNITS[args.units])
    program = plan.model_structure.program
    fn = program.functions["main"]

    seq, by_type = [], Counter()
    for op in fn.block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is None:  # constants and other ops with no compute
            continue
        dev = device_name(usage.preferred_compute_device)
        seq.append(dev)
        by_type[(op.operator_name, dev)] += 1
    totals = Counter(seq)
    switches = sum(a != b for a, b in zip(seq, seq[1:]))
    n = len(seq)
    print(f"{n} compute ops · " + " · ".join(f"{d} {c} ({100 * c / n:.1f}%)" for d, c in totals.most_common())
          + f" · device switches along the graph: {switches}")
    off = Counter({k: v for k, v in by_type.items() if k[1] != "ANE"})
    if off:
        print("not on the Neural Engine:", ", ".join(f"{t}×{c} → {d}" for (t, d), c in off.most_common(12)))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"mlpackage": args.mlpackage, "units": args.units, "ops": n, "by_device": dict(totals),
                       "switches": switches, "off_ane": {f"{t}|{d}": c for (t, d), c in off.items()}}, f, indent=2)


if __name__ == "__main__":
    main()
