"""Bounded, exact time-MLP reuse for a CPU-declared inference schedule.

The sampler must call ``set_step(i)`` before each scheduled model evaluation.
GPU timestep values are never read back. The engine therefore owns the contract
that the declared schedule is the source of the tensors passed to the model.
Misses execute the original callback with its original matrix shapes; stored
outputs are independent copies and must be treated as read-only by consumers.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from contextlib import contextmanager
from numbers import Real
from threading import Lock

import torch


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        # Inference-mode tensors have no version counter. The cache's output
        # contract remains read-only, and its storage never aliases a callback.
        return None


def _tensor_metadata(tensor):
    return tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device


def _module_signature(module):
    if any(child.training for child in module.modules()):
        raise RuntimeError("Time-MLP reuse requires every time module to be in eval mode")
    tensors = tuple(module.named_parameters()) + tuple(module.named_buffers())
    versions = tuple(_version(value) for _, value in tensors)
    if any(version is None for version in versions):
        raise RuntimeError("Time-MLP caching requires versioned parameters and buffers; inference tensors are unsupported")
    return (
        id(module), type(module), type(module).forward, module.__dict__.get("forward"),
        tuple(
            (name, id(child), type(child), child.training, type(child).forward, child.__dict__.get("forward"))
            for name, child in module.named_modules()
        ),
        tuple(
            (name, id(value), value.data_ptr(), version, _tensor_metadata(value))
            for (name, value), version in zip(tensors, versions)
        ),
        getattr(module, "freq_dim", None), getattr(module, "n_params", None),
    )


def _execution_signature(device):
    device = torch.device(device)
    device_type = device.type
    # Storage produced on one asynchronous stream must not become a cache hit
    # on another stream without an explicit dependency. Keep entries separate.
    stream = torch.cuda.current_stream(device) if device_type == "cuda" else None
    execution_device = stream.device if stream is not None else device
    return (
        execution_device, stream.cuda_stream if stream is not None else None,
        torch.is_inference_mode_enabled(),
        torch.is_autocast_enabled(device_type),
        torch.get_autocast_dtype(device_type) if device_type in ("cpu", "cuda") else None,
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        torch.are_deterministic_algorithms_enabled(),
    )


class FixedScheduleTimeCache:
    """Reuse original time embeddings/modulations across matching requests.

    Bind the same instance to both backbones' ``_serving_time_cache``. Enter
    ``generation(schedule, timestep_scales=(video_scale, action_scale),
    precomputed=True)`` around sampling; ``schedule`` includes the terminal
    endpoint. Explicit step indices allow either modality's callback to be
    skipped by velocity/residual fast paths without changing later cache keys.
    """

    def __init__(self, architecture, *, max_entries=32, max_bytes=256 * 1024 * 1024):
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        self.architecture = architecture
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries = OrderedDict()
        self._bytes = 0
        self._lock = Lock()
        self.active = False
        self._schedule_key = None
        self._num_steps = 0
        self._step = -1
        self._called = set()
        self._hits = self._misses = self._evictions = self._invalidations = 0
        self._video_calls = self._action_calls = 0

    @property
    def stats(self):
        return {
            "active": self.active,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "invalidations": self._invalidations,
            "entries": len(self._entries),
            "bytes": self._bytes,
            "steps_completed": self._step + 1,
            "video_calls": self._video_calls,
            "action_calls": self._action_calls,
        }

    def clear(self):
        if self.active:
            raise RuntimeError("Cannot clear the time cache during a generation")
        self._entries.clear()
        self._bytes = 0

    def _validate_evaluation(self):
        if torch.is_grad_enabled() or any(
            module.training for module in (
                self.architecture, self.architecture.video_backbone, self.architecture.action_backbone
            )
        ):
            raise RuntimeError("Time-MLP reuse requires eval-mode, no-grad inference")

    @staticmethod
    def _cpu_schedule(schedule, scales, precomputed):
        def finite_number(value):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                raise ValueError("Time-cache schedule metadata must contain finite CPU numbers, not tensors")
            return float(value)

        if isinstance(schedule, torch.Tensor):
            raise ValueError("Time-cache schedule must come from CPU sampler metadata, not a tensor")
        rows = []
        for row in schedule:
            if isinstance(row, torch.Tensor):
                raise ValueError("Time-cache schedule rows must be CPU numbers, not tensors")
            values = tuple(row)
            if len(values) != 2:
                raise ValueError("Each schedule row must contain video and action timesteps")
            rows.append(tuple(finite_number(value) for value in values))
        scales = tuple(finite_number(value) for value in scales)
        if len(rows) < 2 or len(scales) != 2 or any(scale <= 0 for scale in scales):
            raise ValueError("A schedule needs an update and endpoint, plus two positive timestep scales")
        if not isinstance(precomputed, bool):
            raise ValueError("precomputed must be a boolean")
        return tuple(rows), scales, precomputed

    @contextmanager
    def generation(self, schedule, *, timestep_scales=(1000.0, 1000.0), precomputed=True):
        self._validate_evaluation()
        signature = self._cpu_schedule(schedule, timestep_scales, precomputed)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Time cache does not support concurrent/reentrant generations")
        self._schedule_key = signature
        self._num_steps = len(signature[0]) - 1
        self._step = -1
        self._called.clear()
        self._hits = self._misses = self._evictions = self._invalidations = 0
        self._video_calls = self._action_calls = 0
        self.active = True
        try:
            yield self
            if self._step + 1 != self._num_steps:
                raise RuntimeError("Time cache did not observe every sampler step through set_step")
        finally:
            # Valid, immutable entries can survive a failed request. Only the
            # request's step/schedule binding is discarded, preserving errors.
            self.active = False
            self._schedule_key = None
            self._called.clear()
            self._lock.release()

    def set_step(self, step):
        if not self.active:
            raise RuntimeError("set_step requires an active time-cache generation")
        if isinstance(step, bool) or not isinstance(step, int) or step != self._step + 1:
            raise ValueError("Time-cache steps must advance consecutively from zero")
        if step >= self._num_steps:
            raise ValueError("Time-cache step exceeds the declared schedule")
        self._step = step
        self._called.clear()

    def _get(self, stream, key, compute):
        self._validate_evaluation()
        if not self.active or self._step < 0:
            raise RuntimeError("Time-cache lookup requires generation and an explicit set_step")
        if stream in self._called:
            raise RuntimeError(f"Time-cache {stream} lookup already occurred at this sampler step")
        key = (self._schedule_key, self._step, stream, key)
        entry = self._entries.get(key)
        if entry is not None:
            values, versions, size = entry
            if tuple(_version(value) for value in values) != versions:
                del self._entries[key]
                self._bytes -= size
                self._invalidations += 1
                entry = None
        if entry is None:
            values = compute()
            if not isinstance(values, (tuple, list)) or len(values) != 2 or any(
                not isinstance(value, torch.Tensor) or not value.is_floating_point() for value in values
            ):
                raise TypeError("A time-MLP callback must return two floating-point tensors")
            self._misses += 1
            size = sum(value.numel() * value.element_size() for value in values)
            if size <= self.max_bytes:
                # Detach + clone outside a captured function: graph replay or
                # later mutation of the callback's allocations cannot poison us.
                values = tuple(value.detach().clone() for value in values)
                while self._entries and (
                    len(self._entries) >= self.max_entries or self._bytes + size > self.max_bytes
                ):
                    _, (_, _, removed_size) = self._entries.popitem(last=False)
                    self._bytes -= removed_size
                    self._evictions += 1
                self._entries[key] = values, tuple(_version(value) for value in values), size
                self._bytes += size
            else:
                values = tuple(values)
        else:
            values = entry[0]
            self._entries.move_to_end(key)
            self._hits += 1
        self._called.add(stream)
        if stream == "video":
            self._video_calls += 1
        else:
            self._action_calls += 1
        return values

    def video(
        self, dit, timestep, latents, *, patch_size, fuse_vae_embedding_in_latents,
        force_per_token_t_mod, num_clean_prefix_frames, zero_clean_prefix_t_mod,
        has_first_frame_latents, compute,
    ):
        if dit is not self.architecture.video_backbone.dit:
            raise RuntimeError("Time cache is bound to another video backbone")
        key = (
            _module_signature(dit.time_embedding), _module_signature(dit.time_projection),
            dit.training, dit.seperated_timestep, dit.freq_dim, dit.dim,
            _tensor_metadata(timestep), _tensor_metadata(latents), tuple(patch_size),
            bool(fuse_vae_embedding_in_latents), bool(force_per_token_t_mod),
            int(num_clean_prefix_frames), bool(zero_clean_prefix_t_mod), bool(has_first_frame_latents),
            _execution_signature(latents.device),
        )
        return self._get("video", key, compute)

    def action(self, backbone, timestep, *, compute):
        if backbone is not self.architecture.action_backbone:
            raise RuntimeError("Time cache is bound to another action backbone")
        key = (
            _module_signature(backbone.time_embedding), _module_signature(backbone.time_projection),
            backbone.training, _tensor_metadata(timestep), _execution_signature(timestep.device),
        )
        return self._get("action", key, compute)
