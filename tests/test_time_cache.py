"""CPU checks for exact, schedule-keyed time-MLP serving reuse."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from openwam.deploy.optimizations.time_cache import FixedScheduleTimeCache, _execution_signature
from openwam.model.video_backbone.wan.dit_forward import build_time_modulation


class _DiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.seperated_timestep = True
        self.freq_dim, self.dim = 4, 8
        self.time_embedding = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 8))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(8, 48))


class _ActionTime(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 8)

    def forward(self, timestep):
        return self.linear(timestep[:, None])


def _architecture():
    action = nn.Module()
    action.time_embedding = _ActionTime()
    action.time_projection = nn.Linear(8, 48)
    return SimpleNamespace(
        training=False,
        video_backbone=SimpleNamespace(training=False, dit=_DiT().eval()),
        action_backbone=action.eval(),
    )


_SCHEDULE = ((900.0, 900.0), (450.0, 450.0), (0.0, 0.0))
_VIDEO_KWARGS = {
    "patch_size": (1, 2, 2),
    "fuse_vae_embedding_in_latents": True,
    "force_per_token_t_mod": True,
    "num_clean_prefix_frames": 1,
    "zero_clean_prefix_t_mod": True,
    "has_first_frame_latents": True,
}


def _video(cache, architecture, timestep, *, latents=None, compute=None, **overrides):
    dit = architecture.video_backbone.dit
    if latents is None:
        latents = torch.zeros(1, 2, 3, 4, 4)
    kwargs = dict(_VIDEO_KWARGS, **overrides)
    if compute is None:
        def compute():
            return build_time_modulation(dit, timestep, latents, **kwargs)
    return cache.video(dit, timestep, latents, **kwargs, compute=compute)


def _action(cache, architecture, timestep, *, compute=None):
    backbone = architecture.action_backbone
    if compute is None:
        def compute():
            embedded = backbone.time_embedding(timestep)
            return embedded, backbone.time_projection(embedded)
    return cache.action(backbone, timestep, compute=compute)


def _run(cache, architecture, *, schedule=_SCHEDULE, scales=(1000.0, 1000.0), precomputed=True, **kwargs):
    result = []
    with torch.no_grad(), cache.generation(schedule, timestep_scales=scales, precomputed=precomputed):
        for step, (video_t, action_t) in enumerate(schedule[:-1]):
            cache.set_step(step)
            result.append((_video(cache, architecture, torch.tensor([video_t]), **kwargs),
                           _action(cache, architecture, torch.tensor([action_t]))))
    return result


def test_warm_reuse_is_bitwise_and_keeps_original_360_row_mlp():
    architecture = _architecture()
    cache = FixedScheduleTimeCache(architecture)
    shapes = []
    module = architecture.video_backbone.dit.time_embedding[0]
    hook = module.register_forward_pre_hook(lambda _, args: shapes.append(tuple(args[0].shape)))
    latents = torch.zeros(1, 2, 3, 20, 24)
    cold = _run(cache, architecture, latents=latents)
    assert shapes == [(360, 4), (360, 4)]
    assert cache.stats["misses"] == 4 and cache.stats["hits"] == 0
    warm = _run(cache, architecture, latents=latents)
    assert shapes == [(360, 4), (360, 4)]
    assert cache.stats["hits"] == 4 and cache.stats["misses"] == 0
    for cold_step, warm_step in zip(cold, warm):
        for cold_stream, warm_stream in zip(cold_step, warm_step):
            for expected, actual in zip(cold_stream, warm_stream):
                assert torch.equal(expected, actual)
                assert actual is expected
    # Check against the unmodified callback with the same matrix shapes.
    with torch.no_grad():
        expected = build_time_modulation(
            architecture.video_backbone.dit, torch.tensor([900.0]), latents, **_VIDEO_KWARGS
        )
    assert all(torch.equal(a, b) for a, b in zip(expected, warm[0][0]))
    hook.remove()


def test_explicit_steps_allow_skipped_modalities_without_reindexing():
    architecture = _architecture()
    cache = FixedScheduleTimeCache(architecture)
    _run(cache, architecture)
    with torch.no_grad(), cache.generation(_SCHEDULE):
        cache.set_step(0)
        # A velocity cache can skip both streams at the first sampler step.
        cache.set_step(1)
        _action(cache, architecture, torch.tensor([450.0]), compute=lambda: pytest.fail("expected a hit"))
    assert cache.stats["hits"] == 1
    assert cache.stats["steps_completed"] == 2
    assert cache.stats["video_calls"] == 0 and cache.stats["action_calls"] == 1


@pytest.mark.parametrize("change", ["schedule", "endpoint", "scales", "precomputed"])
def test_sampler_source_metadata_changes_do_not_hit(change):
    architecture = _architecture()
    cache = FixedScheduleTimeCache(architecture)
    _run(cache, architecture)
    kwargs = {
        "schedule": {"schedule": ((899.0, 899.0), (450.0, 450.0), (0.0, 0.0))},
        "endpoint": {"schedule": ((900.0, 900.0), (450.0, 450.0), (1.0, 1.0))},
        "scales": {"scales": (999.0, 1000.0)},
        "precomputed": {"precomputed": False},
    }[change]
    _run(cache, architecture, **kwargs)
    assert cache.stats["misses"] == 4


@pytest.mark.parametrize("kwargs", [
    {"latents": torch.zeros(1, 2, 4, 4, 4)},
    {"latents": torch.zeros(1, 2, 3, 6, 4)},
    {"latents": torch.zeros(1, 2, 3, 4, 6)},
    {"num_clean_prefix_frames": 2},
    {"fuse_vae_embedding_in_latents": False},
    {"force_per_token_t_mod": False},
    {"zero_clean_prefix_t_mod": False},
    {"has_first_frame_latents": False},
    {"patch_size": (1, 1, 2)},
])
def test_video_geometry_and_mode_changes_do_not_hit(kwargs):
    architecture = _architecture()
    cache = FixedScheduleTimeCache(architecture)
    _run(cache, architecture)
    _run(cache, architecture, **kwargs)
    assert cache.stats["misses"] == 2
    assert cache.stats["hits"] == 2  # The independent action stream stays valid.


@pytest.mark.parametrize("kind", ["inplace", "replacement", "module_replacement", "dtype"])
def test_time_weight_changes_invalidate_reuse(kind):
    architecture = _architecture()
    cache = FixedScheduleTimeCache(architecture)
    _run(cache, architecture)
    backbone = architecture.action_backbone
    with torch.no_grad():
        if kind == "inplace":
            backbone.time_projection.weight.add_(1)
        elif kind == "replacement":
            backbone.time_projection.weight = nn.Parameter(backbone.time_projection.weight.detach().clone())
        elif kind == "module_replacement":
            backbone.time_projection = nn.Linear(8, 48).eval()
        else:
            backbone.double()
    with torch.no_grad(), cache.generation(_SCHEDULE):
        for step, (_, action_t) in enumerate(_SCHEDULE[:-1]):
            cache.set_step(step)
            timestep = torch.tensor([action_t], dtype=torch.float64 if kind == "dtype" else torch.float32)
            _action(cache, architecture, timestep)
    assert cache.stats["misses"] == 2


def test_storage_does_not_alias_callback_and_detects_consumer_mutation():
    architecture = _architecture()
    cache = FixedScheduleTimeCache(architecture)
    schedule = _SCHEDULE[:2]
    source = (torch.tensor([[2.0]]), torch.tensor([[3.0]]))
    with torch.no_grad(), cache.generation(schedule):
        cache.set_step(0)
        cached = _video(cache, architecture, torch.tensor([900.0]), compute=lambda: source)
        source[0].fill_(99)
    with torch.no_grad(), cache.generation(schedule):
        cache.set_step(0)
        hit = _video(cache, architecture, torch.tensor([900.0]), compute=lambda: pytest.fail("hit expected"))
        assert hit[0].item() == 2
        assert hit[0].data_ptr() != source[0].data_ptr()
        cached[0].fill_(66)
    with torch.no_grad(), cache.generation(schedule):
        cache.set_step(0)
        actual = _video(cache, architecture, torch.tensor([900.0]), compute=lambda: source)
        assert actual[0].item() == 99
        assert cache.stats["invalidations"] == 1
        assert cache.stats["misses"] == 1


def test_lru_and_byte_budgets_are_bounded():
    architecture = _architecture()
    cache = FixedScheduleTimeCache(architecture, max_entries=2)
    _run(cache, architecture)
    assert cache.stats["entries"] == 2 and cache.stats["evictions"] == 2
    tiny = FixedScheduleTimeCache(architecture, max_bytes=1)
    _run(tiny, architecture)
    assert tiny.stats["entries"] == tiny.stats["bytes"] == 0
    assert tiny.stats["misses"] == 4
    cache.clear()
    assert cache.stats["entries"] == cache.stats["bytes"] == 0


def test_unbound_or_duplicate_steps_fail_and_exception_cleanup_preserves_error():
    architecture = _architecture()
    cache = FixedScheduleTimeCache(architecture)
    with torch.no_grad(), pytest.raises(RuntimeError, match="explicit set_step"):
        with cache.generation(_SCHEDULE):
            _video(cache, architecture, torch.tensor([900.0]))
    assert not cache.active
    with torch.no_grad(), pytest.raises(ValueError, match="consecutively"):
        with cache.generation(_SCHEDULE):
            cache.set_step(1)
    with torch.no_grad(), pytest.raises(RuntimeError, match="already occurred"):
        with cache.generation(_SCHEDULE):
            cache.set_step(0)
            _video(cache, architecture, torch.tensor([900.0]))
            _video(cache, architecture, torch.tensor([900.0]))
    with torch.no_grad(), pytest.raises(RuntimeError, match="model failed"):
        with cache.generation(_SCHEDULE):
            cache.set_step(0)
            raise RuntimeError("model failed")
    assert not cache.active and cache._schedule_key is None
    _run(cache, architecture)


def test_incomplete_normal_generation_and_reentrant_generation_rejected():
    architecture = _architecture()
    cache = FixedScheduleTimeCache(architecture)
    with torch.no_grad(), pytest.raises(RuntimeError, match="every sampler step"):
        with cache.generation(_SCHEDULE):
            cache.set_step(0)
    with torch.no_grad(), pytest.raises(RuntimeError, match="concurrent/reentrant"):
        with cache.generation(_SCHEDULE), cache.generation(_SCHEDULE):
            pass
    assert not cache.active


@pytest.mark.parametrize("schedule", [torch.tensor(_SCHEDULE), ((torch.tensor(1.0), 1.0), (0.0, 0.0)),
                                       ((float("nan"), 1.0), (0.0, 0.0)), ((1.0,), (0.0, 0.0)), ((0.0, 0.0),)])
def test_non_cpu_or_invalid_schedule_rejected(schedule):
    cache = FixedScheduleTimeCache(_architecture())
    with torch.no_grad(), pytest.raises(ValueError):
        with cache.generation(schedule):
            pass


def test_grad_training_and_binding_guards():
    architecture = _architecture()
    cache = FixedScheduleTimeCache(architecture)
    with pytest.raises(RuntimeError, match="no-grad"):
        with cache.generation(_SCHEDULE):
            pass
    architecture.training = True
    with torch.no_grad(), pytest.raises(RuntimeError, match="eval-mode"):
        with cache.generation(_SCHEDULE):
            pass
    architecture.training = False
    with torch.no_grad(), pytest.raises(RuntimeError, match="another action"):
        with cache.generation(_SCHEDULE):
            cache.set_step(0)
            cache.action(_architecture().action_backbone, torch.tensor([900.0]), compute=lambda: None)


def test_autocast_mode_is_part_of_cache_key():
    architecture = _architecture()
    cache = FixedScheduleTimeCache(architecture)
    _run(cache, architecture)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _run(cache, architecture)
    assert cache.stats["misses"] == 4


@pytest.mark.parametrize("kind", ["parameter", "buffer"])
def test_unversioned_time_weights_or_buffers_rejected(kind):
    architecture = _architecture()
    with torch.inference_mode():
        if kind == "parameter":
            architecture.action_backbone.time_projection.weight = nn.Parameter(torch.ones(48, 8))
        else:
            architecture.action_backbone.time_projection.register_buffer("test_buffer", torch.ones(1))
    cache = FixedScheduleTimeCache(architecture)
    with torch.no_grad(), pytest.raises(RuntimeError, match="versioned parameters and buffers"):
        with cache.generation(_SCHEDULE):
            cache.set_step(0)
            _action(cache, architecture, torch.tensor([900.0]))
    assert not cache.active and cache.stats["entries"] == 0


def test_cuda_device_and_stream_are_in_signature_without_gpu_work(monkeypatch):
    stream_handle = [11]
    calls = []

    def current_stream(device):
        calls.append(device)
        return SimpleNamespace(device=device, cuda_stream=stream_handle[0])

    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    first = _execution_signature(torch.device("cuda:0"))
    stream_handle[0] = 12
    other_stream = _execution_signature(torch.device("cuda:0"))
    other_device = _execution_signature(torch.device("cuda:1"))
    assert first != other_stream != other_device
    assert calls == [torch.device("cuda:0"), torch.device("cuda:0"), torch.device("cuda:1")]
    _execution_signature(torch.device("cpu"))
    assert len(calls) == 3


def test_parameterless_time_activation_replacement_invalidates_cache():
    architecture = _architecture()
    cache = FixedScheduleTimeCache(architecture)
    _run(cache, architecture)
    architecture.video_backbone.dit.time_embedding[1] = nn.GELU().eval()
    _run(cache, architecture)
    assert cache.stats["misses"] == 2 and cache.stats["hits"] == 2
