"""Generate explicit cumulative configurations; optionally run isolated paired benchmarks.

No timing is performed with --prepare-only. Each stage runs in a new process;
the common eager baseline is retained for cross-stage output comparisons.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
METHOD_FILE = Path(__file__).with_name("ablation_methods.json")
CUMULATIVE = [
    "compile", "conditional_kv", "compact_context", "context_embedding",
    "split_attention", "static_actions", "skip_eval", "batch_timesteps",
    "parallel_images", "prefix", "residual3f", "time_cache", "rope_cache", "lightweight",
]
DEPENDENCIES = {
    "compact_context": {"conditional_kv"}, "context_embedding": {"conditional_kv"},
    "time_cache": {"batch_timesteps"}, "lightweight": {"residual3f"},
    "memoize_similarity": {"velocity"}, "max_autotune": {"compile"},
}


def validate(methods):
    selected = set(methods)
    for method in methods:
        missing = DEPENDENCIES.get(method, set()) - selected
        if missing:
            raise ValueError(f"{method} requires explicitly selecting {sorted(missing)}")
    for pair in ({"velocity", "residual3f"}, {"triton_rope", "triton_norm_rope"}):
        if pair <= selected:
            raise ValueError(f"Mutually exclusive methods: {sorted(pair)}")


def phase_for(methods):
    chosen = set(methods)
    phase = "residual3F" if "residual3f" in chosen else "v3_velocity" if "velocity" in chosen else "full10F"
    if "prefix" in chosen:
        phase = "prefix_" + phase
    for method, prefix in reversed([("time_cache", "time"), ("rope_cache", "rope"), ("lightweight", "light")]):
        if method in chosen:
            phase = prefix + "_" + phase
    return phase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", help="One custom combination plus the baseline; default runs the cumulative ladder.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--action-contract", choices=["libero_eef10", "generic"], default="libero_eef10")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--warmups", type=int, default=6)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    parser.add_argument("--no-prefix-verify", action="store_true", help="Only after validating fixed observation/conditioning invariants.")
    args = parser.parse_args()
    definitions = json.loads(METHOD_FILE.read_text())
    if args.methods and set(args.methods) - definitions.keys():
        parser.error(f"Unknown methods. Available: {', '.join(definitions)}")
    if args.samples < 1 or args.warmups < 0 or (args.action_contract == "generic" and args.fixtures is None):
        parser.error("Positive samples, nonnegative warmups, and real fixtures for generic actions are required.")
    combinations = [[], args.methods] if args.methods else [[]] + [CUMULATIVE[:i] for i in range(1, len(CUMULATIVE) + 1)]
    for methods in combinations:
        validate(methods)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    base = OmegaConf.load(ROOT / "configs/inference_ablation/base.yaml")
    plan = {"status": "prepared", "fixed_controls": "warm prompt cache; no video decode; BF16 inherited from checkpoint; 10 updates; 32 actions; identical fixtures/seeds", "stages": []}
    for index, methods in enumerate(combinations):
        name = f"{index:02d}_" + (methods[-1] if methods else "baseline")
        cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
        for method in methods:
            for key, value in definitions[method].items():
                OmegaConf.update(cfg, key, value, force_add=True)
        cfg.checkpoint_path = str(args.checkpoint.resolve())
        cfg.device = args.device
        cfg.optimization.observation_prefix_cache.compile_enabled = "compile" in methods
        cfg.optimization.observation_prefix_cache.verify_invariants = not args.no_prefix_verify
        config_path = args.output_dir / f"{name}.yaml"
        OmegaConf.save(cfg, config_path)
        output = args.output_dir / f"{name}.json"
        command = [sys.executable, str(Path(__file__).with_name("benchmark_realtimewam.py")),
                   "--checkpoint", str(args.checkpoint.resolve()), "--config", str(config_path),
                   "--output", str(output), "--phases", phase_for(methods),
                   "--device", args.device, "--action-contract", args.action_contract,
                   "--samples", str(args.samples), "--warmups", str(args.warmups),
                   "--seeds", *map(str, args.seeds)]
        if "compile" not in methods:
            command.append("--no-compile")
        if args.no_prefix_verify:
            command.append("--no-prefix-verify")
        if args.fixtures:
            command.extend(["--fixtures", str(args.fixtures.resolve())])
        plan["stages"].append({"name": name, "methods": methods, "config": str(config_path), "output": str(output), "command": command})
    plan_file = args.output_dir / "plan.json"
    plan_file.write_text(json.dumps(plan, indent=2) + "\n")
    if args.prepare_only:
        print(f"Prepared {len(combinations)} configurations; no benchmark executed: {plan_file}")
        return
    summary = []
    baseline = None
    for stage in plan["stages"]:
        stage["status"] = "running"
        plan_file.write_text(json.dumps(plan, indent=2) + "\n")
        try:
            subprocess.run(stage["command"], cwd=ROOT, check=True)
            result = json.loads(Path(stage["output"]).read_text())
            phase = result["phases"][0]
            if result["status"] != "complete" or phase["status"] != "complete":
                raise RuntimeError("Incomplete benchmark result")
            with np.load(phase["arrays"], allow_pickle=False) as arrays:
                norm, raw = arrays["paired_normalized"].copy(), arrays["paired_raw"].copy()
            signature = (result["cases"], result["fixtures"], result["active_normalized_indices"])
            if baseline is None:
                baseline = norm, raw, signature, phase["latency_ms"]["mean"]
            if signature != baseline[2] or norm.shape != baseline[0].shape or raw.shape != baseline[1].shape:
                raise RuntimeError("Cross-stage fixture/seed/action contract mismatch")
            active = result["active_normalized_indices"]
            delta = np.abs(norm[..., active].astype(np.float64) - baseline[0][..., active])
            summary.append({"stage": stage["name"], "methods": stage["methods"],
                            "latency_ms": phase["latency_ms"],
                            "speedup_vs_baseline": baseline[3] / phase["latency_ms"]["mean"],
                            "active_mae_vs_baseline": float(delta.mean()),
                            "active_max_abs_vs_baseline": float(delta.max()),
                            "raw_mae_vs_baseline": float(np.abs(raw.astype(np.float64) - baseline[1]).mean()),
                            "normalized_bitwise_vs_baseline": bool(np.array_equal(norm, baseline[0]))})
            stage["status"] = "complete"
            (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            rows = ["| Stage | Mean ms | P95 ms | Speedup | Active MAE vs common baseline |",
                    "|---|---:|---:|---:|---:|"]
            rows.extend(f"| {r['stage']} | {r['latency_ms']['mean']:.3f} | {r['latency_ms']['p95']:.3f} | {r['speedup_vs_baseline']:.3f} | {r['active_mae_vs_baseline']:.6g} |" for r in summary)
            (args.output_dir / "summary.md").write_text("\n".join(rows) + "\n")
        except Exception:
            stage["status"] = "failed"
            plan["status"] = "failed"
            raise
        finally:
            plan_file.write_text(json.dumps(plan, indent=2) + "\n")
    plan["status"] = "complete"
    plan_file.write_text(json.dumps(plan, indent=2) + "\n")


if __name__ == "__main__":
    main()
