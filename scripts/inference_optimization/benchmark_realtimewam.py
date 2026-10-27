"""Paired complete-request latency and action-deviation probes for RealtimeWAM.

No simulator or success-rate claim: the reference is the same model's dense
10-forward output. LIBERO EEF10 XYZ values are normalized OSC delta commands,
not metres. All comparisons keep ten scheduler updates, input, seed and weights.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from openwam.deploy.optimizations.dit_cache import DiTVelocityCache  # noqa: E402
from openwam.deploy.server import build_server_from_config  # noqa: E402
from scripts.inference_optimization.benchmark_utils import compilation_state, describe, observation  # noqa: E402

SCHEDULES = {
    "residual10F": tuple(range(10)),
    "residual6F": (0, 2, 4, 6, 8, 9),
    "residual4F": (0, 3, 6, 9),
    "residual2F": (0, 3),
    "residual2F_last": (0, 9),
    "residual3F": (0, 5, 9),
    "residual3F_late": (0, 7, 9),
    "residual4F_late": (0, 4, 7, 9),
    "residual5F": (0, 3, 6, 8, 9),
}
BASE_PHASES = ("full10F", "v3_velocity", *SCHEDULES)
PHASES = (
    *BASE_PHASES, *("prefix_" + name for name in BASE_PHASES),
    "rope_prefix_residual3F", "time_prefix_residual3F", "light_prefix_residual3F",
    "time_rope_prefix_residual3F", "time_rope_light_prefix_residual3F",
    "time_rope_full10F", "time_rope_prefix_residual4F", "time_rope_light_prefix_residual4F",
)


def parse_phase(name):
    options = set()
    while name.split("_", 1)[0] in {"time", "rope", "light"}:
        option, name = name.split("_", 1)
        options.add(option)
    return name, options


def error_summary(delta):
    value = np.asarray(delta, dtype=np.float64)
    absolute = np.abs(value).reshape(-1)
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(value * value))),
        "p95_abs": float(np.percentile(absolute, 95)),
        "max_abs": float(absolute.max()),
    }


def compare_outputs(actual, reference, active_indices, prefix, *, libero_metrics=True):
    norm, raw = actual
    norm_ref, raw_ref = reference
    if norm.shape != norm_ref.shape or raw.shape != raw_ref.shape:
        raise ValueError("Reference and candidate output shapes differ.")
    delta = norm - norm_ref
    raw_delta = raw - raw_ref
    result = {
        "unified_normalized": error_summary(delta),
        "active_normalized": error_summary(delta[..., active_indices]),
        "active_normalized_prefix": error_summary(delta[:prefix, active_indices]),
        "raw_command": error_summary(raw_delta),
        "raw_command_prefix": error_summary(raw_delta[:prefix]),
        "normalized_bitwise_equal": bool(np.array_equal(norm, norm_ref)),
        "raw_bitwise_equal": bool(np.array_equal(raw, raw_ref)),
    }
    if libero_metrics and raw.shape[-1] == 10:
        result.update(
            {
                "xyz_osc_command": error_summary(raw_delta[..., :3]),
                "xyz_osc_command_l2": describe(np.linalg.norm(raw_delta[..., :3], axis=-1)),
                "xyz_osc_command_prefix": error_summary(raw_delta[:prefix, :3]),
                "rotation6d_command": error_summary(raw_delta[..., 3:9]),
                "gripper_command": error_summary(raw_delta[..., 9]),
                "gripper_disagreement_fraction": float(np.mean((raw[..., 9] > 0) != (raw_ref[..., 9] > 0))),
                "gripper_prefix_disagreement_fraction": float(
                    np.mean((raw[:prefix, 9] > 0) != (raw_ref[:prefix, 9] > 0))
                ),
            }
        )
    return result


def load_fixtures(path, limit):
    """Accept [{observation, id?, seed?}] or direct server observation dicts.

    Explicit image path form is {"images": {camera: {"path": "x.jpg"}}}.
    Relative paths resolve beside the JSON. Bare image strings remain base64.
    """
    if path is None:
        items = []
        for i, prompt in enumerate(
            ("robot picks up the red bottle from the table", "place the blue block into the bowl", "open the drawer")
        ):
            obs = observation()
            obs["state"] = [0.01 * i, -0.01 * i, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, float(i % 2)]
            obs["prompt"] = prompt
            items.append({"id": f"synthetic_{i}", "observation": obs})
        provenance = {"kind": "synthetic", "warning": "Random RGB fixtures are speed/debug inputs, not representative robot scenes."}
    else:
        loaded = json.loads(path.read_text())
        items = loaded.get("fixtures") if isinstance(loaded, dict) else loaded
        if not isinstance(items, list):
            raise ValueError("Fixture JSON must be a list or an object with a fixtures list.")
        provenance = {"kind": "provided", "path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if isinstance(loaded, dict):
            provenance["metadata"] = {key: value for key, value in loaded.items() if key != "fixtures"}
    fixtures = []
    for index, item in enumerate(items[:limit] if limit else items):
        obs = copy.deepcopy(item.get("observation", item))
        fixture_id = item.get("id", f"fixture_{index}")
        fixture_seed = item.get("seed")
        obs.pop("id", None)
        obs.pop("seed", None)
        if "images" not in obs or "prompt" not in obs:
            raise ValueError(f"Fixture {fixture_id} needs images and prompt.")
        for camera, value in obs["images"].items():
            if isinstance(value, dict) and "path" in value:
                image_path = Path(value["path"])
                if not image_path.is_absolute():
                    if path is None:
                        raise ValueError("Relative image paths need a fixture JSON path.")
                    image_path = path.parent / image_path
                obs["images"][camera] = base64.b64encode(image_path.read_bytes()).decode("ascii")
        digest = hashlib.sha256(json.dumps(obs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        fixtures.append({"id": fixture_id, "observation": obs, "seed": fixture_seed, "sha256": digest})
    if not fixtures:
        raise ValueError("No fixtures selected.")
    return fixtures, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/deploy_realtimewam_3f_v2.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--action-contract", choices=["libero_eef10", "generic"], default="libero_eef10",
                        help="generic requires real fixtures and reports dimension-agnostic errors without LIBERO units.")
    parser.add_argument("--fixture-limit", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--phases", nargs="+", choices=PHASES, default=list(BASE_PHASES[:6]))
    parser.add_argument("--reference-phase", choices=PHASES, default="full10F", help="Explicit paired-output reference; non-dense choices measure implementation equivalence only.")
    parser.add_argument("--no-prefix-verify", action="store_true", help="Skip repeated tensor equality checks only after invariant validation.")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=6)
    parser.add_argument("--prefix", type=int, default=8, help="Report first-N-action proxy separately; does not change model execution.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--profile", action="store_true", help="One extra untimed trace per selected phase.")
    args = parser.parse_args()
    if args.action_contract == "generic" and args.fixtures is None:
        parser.error("--action-contract generic requires --fixtures matching the checkpoint")
    if args.samples <= 0 or args.warmups < 0 or args.prefix < 1 or args.fixture_limit < 0:
        parser.error("samples/prefix must be positive; warmups/fixture-limit must be nonnegative.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays_dir = args.output.with_suffix("").with_name(args.output.stem + "-arrays")
    if args.output.exists() or arrays_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output} or {arrays_dir}")
    arrays_dir.mkdir()
    os.environ["WAM_PROFILE"] = "0"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    fixtures, provenance = load_fixtures(args.fixtures, args.fixture_limit)
    cases = [
        {"fixture": i, "seed": seed, "id": f"{fixture['id']}_seed{seed}"}
        for i, fixture in enumerate(fixtures)
        for seed in ([fixture["seed"]] if fixture["seed"] is not None else args.seeds)
    ]
    cfg = OmegaConf.load(args.config)
    compile_cfg = OmegaConf.create(OmegaConf.to_container(cfg.optimization.compile, resolve=True))
    compile_cfg.enabled = not args.no_compile
    cfg.optimization.compile.enabled = False
    cfg.optimization.dit_cache.enabled = False
    OmegaConf.update(cfg, "optimization.residual_cache.enabled", False, force_add=True)
    OmegaConf.update(cfg, "optimization.observation_prefix_cache.enabled", False, force_add=True)
    for option in ("rope_cache", "fixed_schedule_time_cache", "lightweight_residual_reuse"):
        OmegaConf.update(cfg, f"optimization.{option}.enabled", False, force_add=True)
    cfg.optimization.decode_video = False
    cfg.inference.denoise_steps = 10
    cfg.inference.denoise_mode = "sync"
    cfg.inference.inference_mode = "sync"
    results = {
        "status": "initializing",
        "fixture_provenance": provenance,
        "fixtures": [{key: value for key, value in f.items() if key != "observation"} for f in fixtures],
        "cases": cases,
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(torch.device(args.device)),
        "timing": "Synchronized wall-clock server.predict; fresh 32-action chunk; includes decode/preprocess/VAE/conditioning/sampling/postprocess; excludes payload deepcopy, reset, warmup, model load, compilation, network, disk writes and extra profiler requests.",
        "reference_phase": args.reference_phase,
        "reference": (
            "Same execution engine with both approximate caches disabled and all ten full Transformer evaluations."
            if args.reference_phase == "full10F" else
            f"Implementation-equivalence reference: {args.reference_phase}. This is not a dense-model quality reference."
        ),
        "compile_enabled": not args.no_compile,
        "action_contract": args.action_contract,
        "quality_scope": "Action-deviation proxy only, not task success. LIBERO-specific command metrics apply only to libero_eef10. Generic raw errors use the checkpoint action units.",
        "prefix_actions": args.prefix,
        "phases": [],
    }

    def save():
        temp = args.output.with_suffix(args.output.suffix + ".tmp")
        temp.write_text(json.dumps(results, indent=2, default=str) + "\n")
        temp.replace(args.output)

    save()
    server = None
    try:
        start = time.perf_counter()
        server = build_server_from_config(cfg, str(args.checkpoint), device=args.device)
        results["load_seconds"] = time.perf_counter() - start
        engine = server.engine
        engine._profile = False
        arch = engine.architecture
        server._init_policy()
        indices = getattr(arch.normalizer, "_dst_index", None)
        if indices is None:
            raise ValueError("Expected an action-unify normalizer with explicit active indices.")
        active = np.asarray(indices, dtype=np.int64)
        results["active_normalized_indices"] = active.tolist()
        results["model_contract"] = getattr(arch, "repr_contract", None)
        results["inference"] = OmegaConf.to_container(server.cfg.inference, resolve=True)
        arch.apply_compile_optimizations(compile_cfg)
        latest = {}
        current = {"seed": 42, "forward_calls": 0}
        unnormalize = arch.normalizer.unnormalize
        original_generate = engine.generate
        original_forward = arch.forward

        def capture_normalized(actions):
            # Retain the CPU input reference; copy only after the request timer.
            latest["normalized"] = actions
            return unnormalize(actions)

        def capture_generate(conditions):
            conditions = dict(conditions)
            conditions["seed"] = current["seed"]
            answer = original_generate(conditions)
            latest["raw"] = answer["actions"]
            return answer

        def count_forward(*forward_args, **forward_kwargs):
            current["forward_calls"] += 1
            return original_forward(*forward_args, **forward_kwargs)

        arch.normalizer.unnormalize = capture_normalized
        engine.generate = capture_generate
        arch.forward = count_forward

        prefix_runner = None
        runtime_optimizations = {}

        def configure(name):
            nonlocal prefix_runner
            name, options = parse_phase(name)
            engine._dit_cache = None
            engine._residual_cache = None
            arch._serving_residual_cache = None
            engine._prefix_cache = None
            arch._serving_prefix_cache = None
            engine._rope_cache = None
            engine._time_cache = None
            engine._lightweight_reuse = None
            arch._serving_time_cache = None
            arch._serving_lightweight_reuse = None
            arch.video_backbone._serving_rope_cache = None
            arch.video_backbone._serving_time_cache = None
            arch.action_backbone._serving_time_cache = None
            if "rope" in options:
                from openwam.deploy.optimizations.rope_cache import WanRoPECache

                if "rope" not in runtime_optimizations:
                    runtime_optimizations["rope"] = WanRoPECache(max_entries=1)
                engine._rope_cache = runtime_optimizations["rope"]
                arch.video_backbone._serving_rope_cache = engine._rope_cache
            if "time" in options:
                from openwam.deploy.optimizations.time_cache import FixedScheduleTimeCache

                if "time" not in runtime_optimizations:
                    runtime_optimizations["time"] = FixedScheduleTimeCache(arch)
                engine._time_cache = runtime_optimizations["time"]
                arch._serving_time_cache = engine._time_cache
                arch.video_backbone._serving_time_cache = engine._time_cache
                arch.action_backbone._serving_time_cache = engine._time_cache
            if "light" in options:
                from openwam.deploy.optimizations.lightweight_reuse import LightweightResidualReuse

                if "light" not in runtime_optimizations:
                    runtime_optimizations["light"] = LightweightResidualReuse(arch)
                engine._lightweight_reuse = runtime_optimizations["light"]
                arch._serving_lightweight_reuse = engine._lightweight_reuse
            if name.startswith("prefix_"):
                from openwam.deploy.optimizations.prefix_cache import ObservationPrefixCache

                if prefix_runner is None:
                    prefix_runner = ObservationPrefixCache(
                        arch, compile_enabled=not args.no_compile,
                        verify_invariants=not args.no_prefix_verify,
                    )
                engine._prefix_cache = prefix_runner
                arch._serving_prefix_cache = prefix_runner
                name = name.removeprefix("prefix_")
            if name == "v3_velocity":
                engine._dit_cache = DiTVelocityCache(cosine_threshold=0.99, max_consecutive_skips=3, memoize_similarity=True)
            elif name in SCHEDULES:
                from openwam.deploy.optimizations.residual_cache import TransformerResidualCache

                cache = TransformerResidualCache(refresh_steps=SCHEDULES[name], total_steps=10)
                engine._residual_cache = cache
                arch._serving_residual_cache = cache

        def request(case_index, phase_name):
            case = cases[case_index]
            payload = copy.deepcopy(fixtures[case["fixture"]]["observation"])
            server.reset()
            latest.clear()
            current.update(seed=case["seed"], forward_calls=0)
            torch.cuda.synchronize(args.device)
            begin = time.perf_counter()
            answer = server.predict(payload)
            torch.cuda.synchronize(args.device)
            elapsed = (time.perf_counter() - begin) * 1000
            norm = np.asarray(latest["normalized"], dtype=np.float32).copy()
            raw_value = latest["raw"]
            if isinstance(raw_value, torch.Tensor):
                raw_value = raw_value.detach().float().cpu().numpy()
            raw = np.asarray(raw_value, dtype=np.float32).copy()
            expected_raw_dim = 10 if args.action_contract == "libero_eef10" else len(active)
            if norm.shape != (32, 80) or raw.shape != (32, expected_raw_dim):
                raise ValueError(f"Expected normalized (32,80), raw (32,{expected_raw_dim}); got {norm.shape}, {raw.shape}")
            if not np.isfinite(norm).all() or not np.isfinite(raw).all():
                raise ValueError("Non-finite model action output")
            if len(server._policy._executor._action_buffer) != 31:
                raise RuntimeError("Request did not generate a new complete 32-action chunk")
            record = {
                "case": case_index,
                "wall_ms": elapsed,
                "server_ms": answer["latency_ms"],
                "forward_calls": current["forward_calls"],
                "normalized_sha256": hashlib.sha256(norm.tobytes()).hexdigest(),
                "raw_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
            }
            if engine._residual_cache is not None:
                stats = dict(engine._residual_cache.stats)
                record["residual_cache"] = stats
                expected = len(SCHEDULES[parse_phase(phase_name)[0].removeprefix("prefix_")])
                if (stats["full_evals"], stats["reused_evals"], stats["head_evals"], stats["steps_completed"]) != (
                    expected, 10 - expected, 10, 10
                ):
                    raise RuntimeError(f"Wrong residual evaluation counts: {stats}")
                if current["forward_calls"] != 10:
                    raise RuntimeError("Residual mode skipped architecture prepare/head calls")
            elif engine._dit_cache is not None:
                stats = dict(engine._dit_cache.stats)
                record["velocity_cache"] = stats
                if stats["total_steps"] != 10 or current["forward_calls"] != 10 - stats["total_skips"]:
                    raise RuntimeError(f"Velocity cache counts disagree: {stats}, {current}")
            elif current["forward_calls"] != 10:
                raise RuntimeError(f"Dense reference ran {current['forward_calls']} forwards")
            if engine._prefix_cache is not None:
                stats = dict(engine._prefix_cache.stats)
                record["prefix_cache"] = stats
                full_evals = record.get("residual_cache", {}).get("full_evals", current["forward_calls"])
                if stats["prefill_evals"] != 1 or stats["joint_evals"] != full_evals:
                    raise RuntimeError(f"Wrong observation-prefix evaluation counts: {stats}")
                if not args.no_compile and not stats.get("compile_active", False):
                    raise RuntimeError(f"Prefix compiled path fell back: {stats}")
            if engine._rope_cache is not None:
                record["rope_cache"] = dict(engine._rope_cache.stats)
            if engine._time_cache is not None:
                stats = dict(engine._time_cache.stats)
                record["time_cache"] = stats
                action_calls = 10 if engine._lightweight_reuse is None else record["residual_cache"]["full_evals"]
                if (stats["steps_completed"], stats["video_calls"], stats["action_calls"]) != (10, 10, action_calls):
                    raise RuntimeError(f"Wrong fixed-schedule time-cache calls: {stats}")
            if engine._lightweight_reuse is not None:
                stats = dict(engine._lightweight_reuse.stats)
                record["lightweight_reuse"] = stats
                if stats["reuse_evals"] != 10 - len(SCHEDULES[parse_phase(phase_name)[0].removeprefix("prefix_")]):
                    raise RuntimeError(f"Lightweight path did not execute all expected reuse steps: {stats}")
            return record, (norm, raw)

        references = {}
        configure(args.reference_phase)
        # Warm every fixture/seed, even when a small timing sample count is used.
        results["status"] = "reference_warmup"
        save()
        for index in range(max(args.warmups, len(cases))):
            record, _ = request(index % len(cases), args.reference_phase)
            print("REFERENCE_WARMUP", index, round(record["wall_ms"], 3), flush=True)
        reference_norm, reference_raw = [], []
        for index in range(len(cases)):
            _, pair = request(index, args.reference_phase)
            references[index] = pair
            reference_norm.append(pair[0])
            reference_raw.append(pair[1])
        ref_file = arrays_dir / "reference.npz"
        np.savez_compressed(ref_file, normalized=np.stack(reference_norm), raw=np.stack(reference_raw), active_indices=active)
        results["reference_arrays"] = str(ref_file)
        results["status"] = "benchmarking"
        save()
        for phase_index, phase_name in enumerate(args.phases):
            configure(phase_name)
            phase = {"name": phase_name, "index": phase_index, "warmups": [], "samples": [], "paired_errors": []}
            results["phases"].append(phase)
            print("PHASE_START", phase_name, flush=True)
            captured = {"warmup_normalized": [], "warmup_raw": [], "sample_normalized": [], "sample_raw": [], "paired_normalized": [], "paired_raw": []}
            phase_arrays = arrays_dir / f"{phase_index:02d}-{phase_name}.npz"

            def save_arrays():
                available = {key: np.stack(value) for key, value in captured.items() if value}
                if available:
                    np.savez_compressed(phase_arrays, **available)
                    phase["arrays"] = str(phase_arrays)

            try:
                for index in range(max(args.warmups, len(cases))):
                    record, pair = request(index % len(cases), phase_name)
                    phase["warmups"].append(record)
                    captured["warmup_normalized"].append(pair[0])
                    captured["warmup_raw"].append(pair[1])
                for index in range(args.samples):
                    record, pair = request(index % len(cases), phase_name)
                    if record.get("time_cache", {}).get("misses", 0):
                        raise RuntimeError(f"Unexpected time-cache miss after warmup: {record['time_cache']}")
                    phase["samples"].append(record)
                    captured["sample_normalized"].append(pair[0])
                    captured["sample_raw"].append(pair[1])
                # Paired probes are outside latency samples and cover every case.
                for index in range(len(cases)):
                    _, pair = request(index, phase_name)
                    captured["paired_normalized"].append(pair[0])
                    captured["paired_raw"].append(pair[1])
                    phase["paired_errors"].append({"case": index, **compare_outputs(pair, references[index], active, args.prefix, libero_metrics=args.action_contract == "libero_eef10")})
                phase["latency_ms"] = describe([row["wall_ms"] for row in phase["samples"]])
                norm_actual = np.stack(captured["paired_normalized"])
                raw_actual = np.stack(captured["paired_raw"])
                # Preserve both aggregate scalar errors and case-level tails.
                phase["aggregate_error"] = {
                    "active_normalized": error_summary(norm_actual[..., active] - np.stack(reference_norm)[..., active]),
                    "active_normalized_prefix": error_summary(norm_actual[:, :args.prefix, active] - np.stack(reference_norm)[:, :args.prefix, active]),
                    "raw_command": error_summary(raw_actual - np.stack(reference_raw)),
                }
                if args.action_contract == "libero_eef10":
                    phase["aggregate_error"].update({
                        "xyz_osc_command": error_summary(raw_actual[..., :3] - np.stack(reference_raw)[..., :3]),
                        "rotation6d_command": error_summary(raw_actual[..., 3:9] - np.stack(reference_raw)[..., 3:9]),
                        "gripper_disagreement_fraction": float(np.mean((raw_actual[..., 9] > 0) != (np.stack(reference_raw)[..., 9] > 0))),
                    })
                phase["compilation"] = compilation_state(arch)
                if not args.no_compile and not phase["compilation"]["mot_compiled_callable_present"]:
                    raise RuntimeError("Compiled MoT silently fell back to eager")
                phase["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
                if args.profile:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
                        request(0, phase_name)
                    trace = args.output.with_name(f"{args.output.stem}-{phase_index:02d}-{phase_name}-trace.json")
                    profiler.export_chrome_trace(str(trace))
                    phase["untimed_profile"] = {
                        "path": str(trace),
                        "cuda_graph_launches": sum(event.count for event in profiler.key_averages() if "cudaGraphLaunch" in event.key),
                    }
                phase["status"] = "complete"
                print("PHASE_COMPLETE", phase_name, json.dumps({"latency_ms": phase["latency_ms"], "error": phase["aggregate_error"]}), flush=True)
            except Exception:
                phase["status"] = "failed"
                phase["error"] = traceback.format_exc()
                raise
            finally:
                save_arrays()
                save()
        results["status"] = "complete"
        save()
        print("REALTIMEWAM_BENCHMARK_COMPLETE", args.output, flush=True)
    except Exception:
        results["status"] = "failed"
        results["error"] = traceback.format_exc()
        save()
        raise
    finally:
        if server is not None:
            server.shutdown()


if __name__ == "__main__":
    main()
