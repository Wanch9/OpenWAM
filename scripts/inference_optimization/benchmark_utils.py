"""Shared benchmark utilities; synthetic observation is LIBERO EEF10 only."""
import base64
import io

import numpy as np
import torch
from PIL import Image


def observation():
    rng = np.random.default_rng(42)

    def image():
        pixels = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    return {
        "images": {"head_camera": image(), "left_wrist_camera": image(), "right_wrist_camera": None},
        "prompt": "robot picks up the red bottle from the table",
        "state": [0.0] * 10,
    }


def describe(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
        "std": float(array.std()),
    }


def compilation_state(architecture):
    from torch._dynamo.utils import counters

    state = {
        "mot_compiled_callable_present": architecture._compiled_mot_run_joint_loop is not None,
        "inductor_counters": dict(counters["inductor"]),
    }
    try:
        from torch._inductor.cudagraph_trees import get_container

        manager = get_container(torch.cuda.current_device()).tree_manager
        state["cuda_graph_tree_manager_present"] = manager is not None
        roots = getattr(manager, "roots", {})
        state["cuda_graph_tree_roots"] = len(roots)
    except Exception as exc:
        state["cuda_graph_diagnostic_error"] = str(exc)
    return state


