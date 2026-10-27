"""Bounded, opt-in cache for Wan's invariant expanded RoPE grid.

The construction order and dtype exactly match ``WanBase.prepare``. Cached
outputs are read-only serving inputs. Entries retain their source axes so an
allocator cannot recycle both Python identity and storage while a key is live.
"""

from __future__ import annotations

from collections import OrderedDict
from threading import RLock

import torch
from torch import Tensor


class WanRoPECache:
    """Reuse one or a few expanded grids during no-grad, eval-mode serving.

    CUDA stream identity is included in the key: a result built asynchronously
    on one stream is never consumed by another stream without synchronization.
    Source inference tensors lacking version counters bypass caching because
    their in-place changes cannot be tracked reliably.
    """

    def __init__(self, *, enabled: bool = True, max_entries: int = 1):
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self.enabled = bool(enabled)
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple, tuple[tuple[Tensor, ...], Tensor]] = OrderedDict()
        self._lock = RLock()
        self._hits = self._misses = self._builds = self._evictions = self._unversioned_bypasses = 0

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "enabled": self.enabled,
                "entries": len(self._entries),
                "max_entries": self.max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "builds": self._builds,
                "evictions": self._evictions,
                "unversioned_bypasses": self._unversioned_bypasses,
            }

    def clear(self) -> None:
        """Drop cached tensors, retaining cumulative diagnostic counters."""
        with self._lock:
            self._entries.clear()

    @staticmethod
    def _axis_signature(axis: Tensor) -> tuple | None:
        try:
            version = axis._version
        except RuntimeError:
            return None
        return (
            id(axis),
            axis.untyped_storage().data_ptr(),
            axis.data_ptr(),
            axis.storage_offset(),
            version,
            tuple(axis.shape),
            tuple(axis.stride()),
            axis.dtype,
            axis.device,
            axis.layout,
        )

    @staticmethod
    def _target_context(device: torch.device) -> tuple[torch.device, int | None]:
        if device.type == "cuda":
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            return device, torch.cuda.current_stream(device).cuda_stream
        return device, None

    @staticmethod
    def _build(freq_axes: tuple[Tensor, ...], grid: tuple[int, int, int], device: torch.device) -> Tensor:
        frames, height, width = grid
        return (
            torch.cat(
                [
                    freq_axes[0][:frames].view(frames, 1, 1, -1).expand(frames, height, width, -1),
                    freq_axes[1][:height].view(1, height, 1, -1).expand(frames, height, width, -1),
                    freq_axes[2][:width].view(1, 1, width, -1).expand(frames, height, width, -1),
                ],
                dim=-1,
            )
            .reshape(frames * height * width, 1, -1)
            .to(device)
        )

    def get(
        self,
        freq_axes,
        grid: tuple[int, int, int],
        device,
        *,
        training: bool,
    ) -> Tensor:
        """Return the original formula's result, sharing storage on cache hits."""
        axes = tuple(freq_axes)
        grid = tuple(grid)
        if len(axes) != 3 or any(not isinstance(axis, Tensor) or axis.ndim != 2 for axis in axes):
            raise ValueError("Wan RoPE requires three two-dimensional frequency axes")
        if len(grid) != 3 or any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in grid):
            raise ValueError("RoPE grid must contain three positive integers")
        if any(axis.shape[0] < size for axis, size in zip(axes, grid)):
            raise ValueError("RoPE grid exceeds a frequency axis length")
        target = torch.device(device)
        if not self.enabled:
            return self._build(axes, grid, target)
        if training or torch.is_grad_enabled():
            raise RuntimeError("Wan RoPE caching requires no-grad eval-mode inference")
        target, stream = self._target_context(target)
        with self._lock:
            signatures = tuple(self._axis_signature(axis) for axis in axes)
            if any(signature is None for signature in signatures):
                self._unversioned_bypasses += 1
                self._builds += 1
                return self._build(axes, grid, target)
            key = (signatures, grid, target, stream)
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                self._hits += 1
                return entry[1]
            self._misses += 1
            value = self._build(axes, grid, target)
            self._builds += 1
            self._entries[key] = (axes, value)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self._evictions += 1
            return value
