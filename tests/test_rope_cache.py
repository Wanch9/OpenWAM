from types import SimpleNamespace

import pytest
import torch

from openwam.deploy.optimizations.rope_cache import WanRoPECache


def axes(dtype=torch.complex128):
    generator = torch.Generator().manual_seed(37)
    return tuple(torch.complex(torch.randn(8, size, generator=generator), torch.randn(8, size, generator=generator)).to(dtype)
                 for size in (3, 2, 2))


def original(freqs, grid):
    frames, height, width = grid
    return torch.cat([
        freqs[0][:frames].view(frames, 1, 1, -1).expand(frames, height, width, -1),
        freqs[1][:height].view(1, height, 1, -1).expand(frames, height, width, -1),
        freqs[2][:width].view(1, 1, width, -1).expand(frames, height, width, -1),
    ], dim=-1).reshape(frames * height * width, 1, -1).to("cpu")


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_original_bits_and_reuse(dtype):
    freqs, grid = axes(dtype), (2, 3, 4)
    cache = WanRoPECache()
    with torch.no_grad():
        first = cache.get(freqs, grid, "cpu", training=False)
        second = cache.get(freqs, grid, torch.device("cpu"), training=False)
    assert torch.equal(first, original(freqs, grid))
    assert first.dtype == dtype
    assert second is first
    assert cache.stats["builds"] == 1 and cache.stats["hits"] == 1


def test_inplace_axis_edit_invalidates_even_when_view_storage_is_shared():
    base = list(axes())
    freqs = (base[0][:], base[1][:], base[2][:])
    cache = WanRoPECache()
    with torch.no_grad():
        first = cache.get(freqs, (2, 3, 4), "cpu", training=False)
        base[1][0, 0] += 1j
        changed = cache.get(freqs, (2, 3, 4), "cpu", training=False)
    assert changed is not first
    assert torch.equal(changed, original(freqs, (2, 3, 4)))
    assert not torch.equal(first, changed)


def test_replacement_storage_and_strides_invalidate():
    freqs = list(axes())
    cache = WanRoPECache(max_entries=3)
    with torch.no_grad():
        first = cache.get(freqs, (2, 3, 4), "cpu", training=False)
        freqs[0] = freqs[0].clone()
        replaced = cache.get(freqs, (2, 3, 4), "cpu", training=False)
        freqs[1] = freqs[1].transpose(0, 1).contiguous().transpose(0, 1)
        strided = cache.get(freqs, (2, 3, 4), "cpu", training=False)
    assert replaced is not first and strided is not replaced
    assert torch.equal(strided, original(freqs, (2, 3, 4)))
    assert cache.stats["builds"] == 3


def test_same_tensor_storage_rebinding_invalidates():
    freqs = list(axes())
    cache = WanRoPECache()
    with torch.no_grad():
        first = cache.get(freqs, (2, 3, 4), "cpu", training=False)
        replacement = freqs[0].clone().add_(2j)
        freqs[0].set_(replacement)
        changed = cache.get(freqs, (2, 3, 4), "cpu", training=False)
    assert changed is not first
    assert torch.equal(changed, original(freqs, (2, 3, 4)))


def test_geometry_lru_bound_and_explicit_clear():
    freqs, cache = axes(), WanRoPECache(max_entries=2)
    with torch.no_grad():
        first = cache.get(freqs, (2, 3, 4), "cpu", training=False)
        cache.get(freqs, (1, 3, 4), "cpu", training=False)
        assert cache.get(freqs, (2, 3, 4), "cpu", training=False) is first
        cache.get(freqs, (3, 3, 4), "cpu", training=False)
        assert cache.stats["entries"] == 2 and cache.stats["evictions"] == 1
        cache.get(freqs, (1, 3, 4), "cpu", training=False)
    assert cache.stats["builds"] == 4
    cache.clear()
    assert cache.stats["entries"] == 0


def test_target_context_is_part_of_cache_key(monkeypatch):
    # Exercise device/stream-context invalidation on CPU, without creating CUDA
    # tensors or relying on accelerator availability in this regression suite.
    context = [0]
    monkeypatch.setattr(WanRoPECache, "_target_context", staticmethod(lambda device: (device, context[0])))
    freqs, cache = axes(), WanRoPECache()
    with torch.no_grad():
        first = cache.get(freqs, (2, 3, 4), "cpu", training=False)
        context[0] = 1
        changed = cache.get(freqs, (2, 3, 4), "cpu", training=False)
    assert changed is not first and torch.equal(changed, first)


def test_training_and_grad_rejected_but_disabled_is_original_path():
    freqs, cache = axes(), WanRoPECache()
    with pytest.raises(RuntimeError, match="no-grad eval"):
        cache.get(freqs, (2, 3, 4), "cpu", training=False)
    with torch.no_grad(), pytest.raises(RuntimeError, match="no-grad eval"):
        cache.get(freqs, (2, 3, 4), "cpu", training=True)
    disabled = WanRoPECache(enabled=False)
    first = disabled.get(freqs, (2, 3, 4), "cpu", training=True)
    second = disabled.get(freqs, (2, 3, 4), "cpu", training=True)
    assert torch.equal(first, original(freqs, (2, 3, 4))) and first is not second
    assert disabled.stats["entries"] == 0


def test_unversioned_inference_axes_bypass_and_observe_mutation():
    cache = WanRoPECache()
    with torch.inference_mode():
        freqs = axes()
        first = cache.get(freqs, (2, 3, 4), "cpu", training=False)
        freqs[0].add_(1j)
        second = cache.get(freqs, (2, 3, 4), "cpu", training=False)
    assert not torch.equal(first, second)
    assert cache.stats["entries"] == 0 and cache.stats["unversioned_bypasses"] == 2


@pytest.mark.parametrize("max_entries", [0, -1, True, 1.5])
def test_invalid_capacity(max_entries):
    with pytest.raises(ValueError, match="positive integer"):
        WanRoPECache(max_entries=max_entries)


@pytest.mark.parametrize("grid", [(0, 1, 1), (True, 1, 1), (1, 2), (9, 1, 1)])
def test_invalid_grid(grid):
    with torch.no_grad(), pytest.raises(ValueError):
        WanRoPECache().get(axes(), grid, "cpu", training=False)


@pytest.mark.parametrize("training,grad_enabled", [(True, True), (True, False), (False, True), (False, False)])
def test_wan_prepare_hook_preserves_training_and_grad_paths(training, grad_enabled):
    from openwam.model.video_backbone.wan.models.dit import WanModel
    from openwam.model.video_backbone.wan_backbone import Wan22Ti2v

    dit = WanModel(
        dim=24, in_dim=2, ffn_dim=48, out_dim=2, text_dim=8, freq_dim=8,
        eps=1e-6, patch_size=(1, 2, 2), num_heads=2, num_layers=1,
        has_image_input=False, seperated_timestep=True,
        require_vae_embedding=False, require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,
    )
    video = Wan22Ti2v(SimpleNamespace(dit=dit), text_dim=8).eval()
    cache = WanRoPECache()
    video._serving_rope_cache = cache
    inputs = {
        "latents": torch.randn(1, 2, 3, 4, 4, requires_grad=grad_enabled),
        "timestep": torch.tensor([700.0]),
        "context": torch.randn(1, 3, 8),
        "context_mask": torch.tensor([[True, True, False]]),
        "fuse_vae_embedding_in_latents": True,
        "num_clean_prefix_frames": 1,
    }
    with torch.no_grad():
        warmed = video.prepare(**inputs)
    assert cache.stats["builds"] == 1
    video.train(training)
    with torch.set_grad_enabled(grad_enabled):
        video._serving_rope_cache = None
        reference = video.prepare(**inputs)
        video._serving_rope_cache = cache
        actual = video.prepare(**inputs)
        assert torch.equal(actual.rope_freqs, reference.rope_freqs)
        assert torch.equal(actual.hidden_states, reference.hidden_states)
        assert torch.equal(actual.time_mod, reference.time_mod)
        if grad_enabled:
            actual.hidden_states.sum().backward()
            assert inputs["latents"].grad is not None
    serving = not training and not grad_enabled
    assert cache.stats["hits"] == int(serving)
    assert cache.stats["builds"] == 1
    assert (actual.rope_freqs is warmed.rope_freqs) == serving
