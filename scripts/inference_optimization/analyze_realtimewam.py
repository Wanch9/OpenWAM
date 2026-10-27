"""Analyze paired RealtimeWAM NPZ outputs without running the model or a simulator.

All quality metrics are deviations from dense 10F predictions. They do not
estimate closed-loop success probabilities. LIBERO XYZ and decoded rotation
vectors are normalized OSC commands; no physical scale is assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.libero.openwam2libero_interface import native_eef10_to_libero7d  # noqa: E402


def distribution(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def absolute_error(actual, reference):
    delta = np.asarray(actual, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    result = distribution(np.abs(delta))
    result["rmse"] = float(np.sqrt(np.mean(delta * delta)))
    return result


def rotation_matrices(rotation6d):
    """Recover first-two-column rotation matrices, marking degenerate commands."""
    r = np.asarray(rotation6d, dtype=np.float64)
    first, second = r[..., :3], r[..., 3:6]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    b1 = first / np.maximum(first_norm, 1e-12)
    orthogonal = second - np.sum(b1 * second, axis=-1, keepdims=True) * b1
    second_norm = np.linalg.norm(orthogonal, axis=-1, keepdims=True)
    b2 = orthogonal / np.maximum(second_norm, 1e-12)
    b3 = np.cross(b1, b2)
    valid = (first_norm[..., 0] > 1e-6) & (second_norm[..., 0] > 1e-6)
    return np.stack((b1, b2, b3), axis=-1), valid


def runtime_commands(raw, *, clip):
    flat = np.asarray(raw, dtype=np.float32).reshape(-1, 10)
    return np.stack([native_eef10_to_libero7d(row, clip=clip) for row in flat]).reshape(*raw.shape[:-1], 7)


def gripper_decisions(commands):
    """Sign proxy: negative=open, positive=close, exact +/-0=neutral.

    The runtime command is continuous. These three categories are an analysis
    proxy, not a claim that the controller binarizes its input.
    """
    return np.sign(np.asarray(commands))


def measure(normalized, raw, ref_normalized, ref_raw, active, xyz_scale, rotation_scale):
    commands = runtime_commands(raw, clip=True)
    ref_commands = runtime_commands(ref_raw, clip=True)
    unclipped = runtime_commands(raw, clip=False)
    ref_unclipped = runtime_commands(ref_raw, clip=False)
    delta = commands - ref_commands
    matrices, valid = rotation_matrices(raw[..., 3:9])
    ref_matrices, ref_valid = rotation_matrices(ref_raw[..., 3:9])
    valid_pair = valid & ref_valid
    relative = np.swapaxes(ref_matrices, -1, -2) @ matrices
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5
    skew = np.stack(
        (relative[..., 2, 1] - relative[..., 1, 2],
         relative[..., 0, 2] - relative[..., 2, 0],
         relative[..., 1, 0] - relative[..., 0, 1]), axis=-1,
    )
    # atan2 avoids an arccos roundoff floor for identical/nearby rotations.
    geodesic = np.degrees(np.arctan2(0.5 * np.linalg.norm(skew, axis=-1), np.clip(cosine, -1.0, 1.0)))
    xyz_l2 = np.linalg.norm(delta[..., :3], axis=-1)
    rotation_l2 = np.linalg.norm(delta[..., 3:6], axis=-1)
    gripper_disagreement = gripper_decisions(commands[..., 6]) != gripper_decisions(ref_commands[..., 6])
    result = {
        "active_normalized": absolute_error(normalized[..., active], ref_normalized[..., active]),
        "raw_eef10": absolute_error(raw, ref_raw),
        "raw_xyz_command": absolute_error(raw[..., :3], ref_raw[..., :3]),
        "raw_rotation6d_command": absolute_error(raw[..., 3:9], ref_raw[..., 3:9]),
        "unclipped_runtime_xyz_command": absolute_error(unclipped[..., :3], ref_unclipped[..., :3]),
        "unclipped_runtime_rotation_vector_command": absolute_error(unclipped[..., 3:6], ref_unclipped[..., 3:6]),
        "runtime_xyz_command": absolute_error(commands[..., :3], ref_commands[..., :3]),
        "runtime_xyz_command_l2": distribution(xyz_l2),
        "runtime_rotation_vector_command": absolute_error(commands[..., 3:6], ref_commands[..., 3:6]),
        "runtime_rotation_vector_command_l2": distribution(rotation_l2),
        "runtime_gripper_command": absolute_error(commands[..., 6], ref_commands[..., 6]),
        "encoded_delta_rotation_geodesic_deg": distribution(geodesic[valid_pair]),
        "degenerate_rotation_count": int(np.count_nonzero(~valid)),
        "reference_degenerate_rotation_count": int(np.count_nonzero(~ref_valid)),
        "gripper_decision_disagreement_fraction": float(np.mean(gripper_disagreement)),
        "gripper_decision_disagreement_count": int(np.count_nonzero(gripper_disagreement)),
        "gripper_neutral_command_count": int(np.count_nonzero(commands[..., 6] == 0.0)),
        "reference_gripper_neutral_command_count": int(np.count_nonzero(ref_commands[..., 6] == 0.0)),
        "action_count": int(np.prod(raw.shape[:-1])),
        "clipped_xyz_component_fraction": float(np.mean(np.abs(unclipped[..., :3]) > 1.0)),
        "clipped_rotation_component_fraction": float(np.mean(np.abs(unclipped[..., 3:6]) > 1.0)),
    }
    if xyz_scale is not None:
        result["controller_equivalent_translation_delta_mm"] = distribution(xyz_l2 * xyz_scale * 1000.0)
    if rotation_scale is not None:
        # This is an axis-angle command-vector difference times an explicitly
        # supplied controller scale, not a measured physical orientation error.
        result["controller_equivalent_rotation_vector_delta_deg"] = distribution(
            rotation_l2 * rotation_scale * 180.0 / np.pi
        )
    return result


def resolve_array_path(raw_path, source, directory):
    original = Path(raw_path)
    candidates = [original, source.parent / original.name]
    if directory is not None:
        candidates.insert(0, directory / original.name)
    candidates.append(source.with_suffix("").with_name(source.stem + "-arrays") / original.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not locate {raw_path}; use --arrays-dir for downloaded archives")


def pareto_frontier(rows):
    """Retain points undominated in latency and three independent proxy errors."""
    accepted = []
    for row in rows:
        if row["degenerate_rotations"] or row["reference_degenerate_rotations"]:
            continue
        vector = np.asarray(row["pareto_vector"], dtype=np.float64)
        if not np.isfinite(vector).all():
            continue
        dominated = any(
            other is not row
            and not other["degenerate_rotations"]
            and not other["reference_degenerate_rotations"]
            and np.all(np.asarray(other["pareto_vector"]) <= vector)
            and np.any(np.asarray(other["pareto_vector"]) < vector)
            for other in rows
        )
        if not dominated:
            accepted.append(row["phase_id"])
    return accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arrays-dir", type=Path)
    parser.add_argument("--prefix", type=int, default=8)
    parser.add_argument("--xyz-scale-m", type=float, help="Optional verified isotropic controller metres per unit clipped XYZ command.")
    parser.add_argument("--rotation-scale-rad", type=float, help="Optional verified isotropic controller radians per unit clipped rotation-vector command.")
    parser.add_argument("--max-active-nmae", type=float)
    parser.add_argument("--max-xyz-command-l2-p95", type=float)
    parser.add_argument("--max-rotation-command-l2-p95", type=float)
    parser.add_argument("--max-gripper-disagreement", type=float, help="Maximum sign-proxy decision disagreement fraction; zero is a separate neutral category.")
    args = parser.parse_args()
    if args.prefix < 1:
        parser.error("prefix must be positive")
    for name in ("xyz_scale_m", "rotation_scale_rad"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error(f"{name} must be finite and positive")
    for name in ("max_active_nmae", "max_xyz_command_l2_p95", "max_rotation_command_l2_p95", "max_gripper_disagreement"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value < 0):
            parser.error(f"{name} must be finite and nonnegative")
    if args.max_gripper_disagreement is not None and args.max_gripper_disagreement > 1:
        parser.error("max_gripper_disagreement must not exceed 1")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    source = json.loads(args.benchmark.read_text())
    reference_path = resolve_array_path(source["reference_arrays"], args.benchmark, args.arrays_dir)
    with np.load(reference_path, allow_pickle=False) as archive:
        ref_norm = archive["normalized"]
        ref_raw = archive["raw"]
        active = archive["active_indices"]
    if ref_norm.ndim != 3 or ref_raw.shape != (*ref_norm.shape[:2], 10):
        raise ValueError("Expected case × horizon × dimension arrays and raw EEF10 actions")
    if not np.isfinite(ref_norm).all() or not np.isfinite(ref_raw).all():
        raise ValueError("Reference contains non-finite values")
    windows = {"first_action": 1, f"prefix_{args.prefix}": min(args.prefix, ref_raw.shape[1]), "full_chunk": ref_raw.shape[1]}
    output = {
        "schema_version": 2,
        "source_benchmark": str(args.benchmark.resolve()),
        "reference_arrays": str(reference_path),
        "reference_phase": source.get("reference_phase", "full10F"),
        "quality_scope": (
            f"Deviation from same-input/same-seed {source.get('reference_phase', 'full10F')} predictions; "
            "no closed-loop accuracy or success-rate inference. A non-full10F reference measures "
            "implementation equivalence, not deviation from the dense model."
        ),
        "units": {
            "raw_xyz": "normalized OSC delta command, not metres",
            "runtime_rotation_vector": "normalized OSC rotation-vector command, not physical radians",
            "encoded_delta_rotation_geodesic_deg": "angular difference between Exp(command rotation vectors), before controller scaling",
            "optional_scaled_metrics": "controller-equivalent command differences only; not actual robot trajectory errors",
            "controller_scale_assumption": "one verified positive isotropic linear scale per modality, applied after runtime clipping",
            "gripper_command": "continuous LIBERO command in [-1,1]; negative=open, positive=close",
            "gripper_decision_proxy": "np.sign(command): -1=open, +1=close, exact +0/-0=neutral; proxy categories only, not runtime binarization",
        },
        "explicit_controller_scales": {"xyz_m": args.xyz_scale_m, "rotation_rad": args.rotation_scale_rad},
        "fixture_provenance": source.get("fixture_provenance"),
        "cases": source["cases"],
        "phases": [],
        "pareto_dimensions": ["latency_p95_ms", "prefix_xyz_command_l2_p95", "prefix_rotation_command_l2_p95", "prefix_gripper_decision_disagreement_fraction"],
        "selection_thresholds": {
            "active_nmae": args.max_active_nmae,
            "xyz_command_l2_p95": args.max_xyz_command_l2_p95,
            "rotation_command_l2_p95": args.max_rotation_command_l2_p95,
            "gripper_decision_disagreement": args.max_gripper_disagreement,
        },
    }
    rows = []
    latency_references = {phase["name"]: phase["latency_ms"]["mean"] for phase in source["phases"] if phase.get("status") == "complete"}
    for phase in source["phases"]:
        if phase.get("status") != "complete":
            continue
        path = resolve_array_path(phase["arrays"], args.benchmark, args.arrays_dir)
        with np.load(path, allow_pickle=False) as archive:
            norm = archive["paired_normalized"]
            raw = archive["paired_raw"]
        if norm.shape != ref_norm.shape or raw.shape != ref_raw.shape:
            raise ValueError(f"Mismatched paired arrays for {phase['name']}")
        if not np.isfinite(norm).all() or not np.isfinite(raw).all():
            raise ValueError(f"Non-finite output in {phase['name']}")
        phase_id = f"{phase['index']:02d}-{phase['name']}"
        record = {
            "phase_id": phase_id,
            "name": phase["name"],
            "arrays": str(path),
            "latency_ms": phase["latency_ms"],
            "windows": {},
            "cases": [],
        }
        for window, count in windows.items():
            record["windows"][window] = measure(norm[:, :count], raw[:, :count], ref_norm[:, :count], ref_raw[:, :count], active, args.xyz_scale_m, args.rotation_scale_rad)
        for index, case in enumerate(source["cases"]):
            count = windows[f"prefix_{args.prefix}"]
            record["cases"].append(
                {
                    "case": index,
                    "id": case["id"],
                    "prefix": measure(norm[index, :count], raw[index, :count], ref_norm[index, :count], ref_raw[index, :count], active, args.xyz_scale_m, args.rotation_scale_rad),
                    "full_chunk_active_normalized": absolute_error(norm[index, :, active], ref_norm[index, :, active]),
                }
            )
        command = runtime_commands(raw, clip=True)
        reference_command = runtime_commands(ref_raw, clip=True)
        record["per_action_step"] = [
            {
                "step": step,
                "active_normalized_mae": float(np.mean(np.abs(norm[:, step, active] - ref_norm[:, step, active]))),
                "xyz_command_l2_mean": float(np.linalg.norm(command[:, step, :3] - reference_command[:, step, :3], axis=-1).mean()),
                "rotation_command_l2_mean": float(np.linalg.norm(command[:, step, 3:6] - reference_command[:, step, 3:6], axis=-1).mean()),
                "gripper_command_mae": float(np.mean(np.abs(command[:, step, 6] - reference_command[:, step, 6]))),
                "gripper_decision_disagreement_fraction": float(np.mean(gripper_decisions(command[:, step, 6]) != gripper_decisions(reference_command[:, step, 6]))),
            }
            for step in range(raw.shape[1])
        ]
        record["worst_prefix_cases"] = [case["id"] for case in sorted(record["cases"], key=lambda c: c["prefix"]["active_normalized"]["mean"], reverse=True)[:5]]
        for reference_name in ("full10F", "v3_velocity"):
            if reference_name in latency_references:
                record[f"mean_speedup_vs_{reference_name}"] = latency_references[reference_name] / phase["latency_ms"]["mean"]
        prefix = record["windows"][f"prefix_{args.prefix}"]
        row = {
            "phase_id": phase_id,
            "name": phase["name"],
            "mean_ms": phase["latency_ms"]["mean"],
            "p95_ms": phase["latency_ms"]["p95"],
            "prefix_active_nmae": prefix["active_normalized"]["mean"],
            "prefix_xyz_command_l2_p95": prefix["runtime_xyz_command_l2"]["p95"],
            "prefix_rotation_command_l2_p95": prefix["runtime_rotation_vector_command_l2"]["p95"],
            "prefix_gripper_command_mae": prefix["runtime_gripper_command"]["mean"],
            "prefix_gripper_decision_disagreement_fraction": prefix["gripper_decision_disagreement_fraction"],
            "degenerate_rotations": prefix["degenerate_rotation_count"],
            "reference_degenerate_rotations": prefix["reference_degenerate_rotation_count"],
        }
        row["pareto_vector"] = [row["p95_ms"], row["prefix_xyz_command_l2_p95"], row["prefix_rotation_command_l2_p95"], row["prefix_gripper_decision_disagreement_fraction"]]
        violations = []
        for field, limit in (
            ("prefix_active_nmae", args.max_active_nmae),
            ("prefix_xyz_command_l2_p95", args.max_xyz_command_l2_p95),
            ("prefix_rotation_command_l2_p95", args.max_rotation_command_l2_p95),
            ("prefix_gripper_decision_disagreement_fraction", args.max_gripper_disagreement),
        ):
            if limit is not None and row[field] > limit:
                violations.append(field)
        if row["degenerate_rotations"]:
            violations.append("degenerate_rotations")
        if row["reference_degenerate_rotations"]:
            violations.append("reference_degenerate_rotations")
        row["threshold_violations"] = violations
        row["passes_explicit_proxy_thresholds"] = not violations if any(value is not None for value in output["selection_thresholds"].values()) else None
        rows.append(row)
        output["phases"].append(record)
    output["comparison"] = rows
    output["pareto_frontier"] = pareto_frontier(rows)
    admissible = [row for row in rows if row["passes_explicit_proxy_thresholds"] is True]
    output["fastest_under_explicit_proxy_thresholds"] = min(admissible, key=lambda row: row["p95_ms"])["phase_id"] if admissible else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"comparison": rows, "pareto_frontier": output["pareto_frontier"], "fastest_under_explicit_proxy_thresholds": output["fastest_under_explicit_proxy_thresholds"]}, indent=2))


if __name__ == "__main__":
    main()
