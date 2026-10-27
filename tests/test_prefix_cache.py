import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from openwam.deploy.optimizations.prefix_cache import ObservationPrefixCache
from openwam.model.video_backbone.wan_backbone import WanBase


class _Video(WanBase):
    @classmethod
    def from_pretrained(cls, source, **kwargs):
        raise NotImplementedError("The test backbone does not load checkpoints")

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.dit = SimpleNamespace(fuse_vae_embedding_in_latents=True, require_clip_embedding=False)
        self.vace = None
        self._video_attention_mask_mode = "first_frame_causal"
        self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 12, bias=False) for _ in range(2)])
        self.rows = []

    def pre_attn_at_layer_for_compile(self, layer, state):
        self.rows.append(state.hidden_states.shape[1])
        h = state.hidden_states + state.time_mod[:, :, 0] + state.rope_freqs[:, 0]
        q, k, v = self.layers[layer](h).chunk(3, dim=-1)
        return q, k, v, state.hidden_states

    def post_attn_at_layer_for_compile(self, layer, state, attended, previous):
        context = state.context
        if state.context_mask is not None:
            context = context * state.context_mask.unsqueeze(-1)
        state.hidden_states = torch.tanh(previous + attended + context.mean(dim=1, keepdim=True) * 0.1)
        return state


class _Action(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 12, bias=False) for _ in range(2)])

    def pre_attn_at_layer_for_compile(self, layer, state):
        h = state.payload.x_action + state.payload.time
        q, k, v = self.layers[layer](h).chunk(3, dim=-1)
        return q, k, v, state.payload.x_action

    def post_attn_at_layer_for_compile(self, layer, state, attended, previous):
        state.payload.x_action = torch.tanh(previous + attended)
        return state


def _setup():
    torch.manual_seed(177)
    vb, ab = _Video().eval(), _Action().eval()
    driver = SimpleNamespace(num_layers=2, num_heads=2, attention_mask_mode="mutual")
    architecture = SimpleNamespace(video_backbone=vb, action_backbone=ab, mot_driver=driver, training=False)
    initial = SimpleNamespace(
        hidden_states=torch.randn(1, 6, 4),
        time_mod=torch.randn(1, 6, 6, 4),
        rope_freqs=torch.randn(6, 1, 4),
        context=torch.randn(1, 3, 4),
        context_mask=torch.tensor([[True, True, False]]),
        grid_frames=3,
        grid_height=1,
        grid_width=2,
        extras={"vace": None},
        vace_hints=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    )
    return architecture, initial


def _states(initial, step):
    video = copy.copy(initial)
    video.hidden_states = initial.hidden_states.clone()
    video.hidden_states[:, 2:] += step * 0.1
    video.time_mod = initial.time_mod.clone()
    video.time_mod[:, 2:] += step * 0.2
    action = SimpleNamespace(payload=SimpleNamespace(x_action=torch.full((1, 3, 4), step * 0.3), time=step * 0.1))
    return video, action


def _dense(architecture, video, action):
    # Independent dense reference: one masked attention over all token rows.
    vb, ab = architecture.video_backbone, architecture.action_backbone
    mask = torch.ones(9, 9, dtype=torch.bool)
    mask[:2, 2:] = False
    for layer in range(2):
        qv, kv, vv, vp = vb.pre_attn_at_layer_for_compile(layer, video)
        qa, ka, va, ap = ab.pre_attn_at_layer_for_compile(layer, action)
        q, k, v = [torch.cat(parts, dim=1).view(1, 9, 2, 2).transpose(1, 2) for parts in ((qv, qa), (kv, ka), (vv, va))]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(1, 9, 4)
        video = vb.post_attn_at_layer_for_compile(layer, video, out[:, :6], vp)
        action = ab.post_attn_at_layer_for_compile(layer, action, out[:, 6:], ap)
    return video.hidden_states, action.payload.x_action


def test_prefix_cache_matches_dense_with_changing_future_and_action():
    architecture, initial = _setup()
    cache = ObservationPrefixCache(architecture, compile_enabled=False)
    with torch.no_grad(), cache.generation():
        for step in range(4):
            expected_v, expected_a = _dense(architecture, *_states(initial, step))
            actual_v, actual_a = cache.run(*_states(initial, step))
            torch.testing.assert_close(actual_v.hidden_states, expected_v, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(actual_a.payload.x_action, expected_a, rtol=1e-5, atol=1e-6)
        assert cache.stats["prefill_evals"] == 1
        assert cache.stats["joint_evals"] == 4
        assert cache.stats["invariant_checks"] == 3
        # Exactly one prefix-only traversal of both layers.
        assert architecture.video_backbone.rows.count(2) == 2
        assert architecture.video_backbone.rows.count(4) == 8
    assert cache._keys is None and cache._prefix_final is None and not cache.active


@pytest.mark.parametrize("field", ["hidden_states", "time_mod", "rope_freqs", "context", "context_mask"])
def test_changed_prefix_condition_rejected(field):
    architecture, initial = _setup()
    cache = ObservationPrefixCache(architecture, compile_enabled=False)
    with torch.no_grad(), pytest.raises(ValueError, match="invariant changed"):
        with cache.generation():
            cache.run(*_states(initial, 0))
            video, action = _states(initial, 1)
            value = getattr(video, field).clone()
            value.reshape(-1)[0] = not bool(value.reshape(-1)[0]) if value.dtype == torch.bool else 999
            setattr(video, field, value)
            cache.run(video, action)
    assert cache._keys is None and not cache.active


def test_new_request_rebuilds_changed_prefix_and_context():
    architecture, initial = _setup()
    cache = ObservationPrefixCache(architecture, compile_enabled=False)
    with torch.no_grad():
        for request in range(2):
            initial.hidden_states += 0.25 * request
            initial.context += 0.5 * request
            with cache.generation():
                expected_v, expected_a = _dense(architecture, *_states(initial, 0))
                actual_v, actual_a = cache.run(*_states(initial, 0))
                torch.testing.assert_close(actual_v.hidden_states, expected_v, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(actual_a.payload.x_action, expected_a, rtol=1e-5, atol=1e-6)
                assert cache.stats["prefill_evals"] == 1


def test_bad_visibility_and_non_ti2v_rejected():
    architecture, _ = _setup()
    architecture.mot_driver.attention_mask_mode = "action_sees_video"
    with pytest.raises(ValueError, match="mutual"):
        ObservationPrefixCache(architecture, compile_enabled=False)
    architecture.mot_driver.attention_mask_mode = "mutual"
    architecture.video_backbone.dit.fuse_vae_embedding_in_latents = False
    with pytest.raises(ValueError, match="TI2V"):
        ObservationPrefixCache(architecture, compile_enabled=False)


@pytest.mark.parametrize("kind", ["gradient", "training", "time_mod", "extra_kv", "shape", "vace"])
def test_invalid_execution_states_rejected(kind):
    architecture, initial = _setup()
    cache = ObservationPrefixCache(architecture, compile_enabled=False)
    video, action = _states(initial, 0)
    if kind == "training":
        architecture.training = True
    elif kind == "time_mod":
        video.time_mod = video.time_mod[:, 0]
    elif kind == "extra_kv":
        video.prefix_kv_len = 3
    elif kind == "shape":
        video.hidden_states = video.hidden_states[:, :-1]
    elif kind == "vace":
        video.vace_hints = [torch.ones(1)]
    context = torch.enable_grad() if kind == "gradient" else torch.no_grad()
    with context, pytest.raises((RuntimeError, ValueError)):
        with cache.generation():
            cache.run(video, action)
    assert not cache.active and cache._keys is None


def test_exception_and_reentrant_cleanup():
    architecture, initial = _setup()
    cache = ObservationPrefixCache(architecture, compile_enabled=False)
    with torch.no_grad(), pytest.raises(RuntimeError, match="simulated"):
        with cache.generation():
            cache.run(*_states(initial, 0))
            with pytest.raises(RuntimeError, match="reentrant"):
                with cache.generation():
                    pass
            raise RuntimeError("simulated")
    assert not cache.active and cache._keys is None
    with torch.no_grad(), cache.generation():
        cache.run(*_states(initial, 1))


def test_empty_successful_generation_rejected():
    architecture, _ = _setup()
    cache = ObservationPrefixCache(architecture, compile_enabled=False)
    with pytest.raises(RuntimeError, match="did not run"):
        with cache.generation():
            pass
    assert not cache.active and cache._keys is None


def test_compile_failure_restarts_from_unmodified_states():
    architecture, initial = _setup()
    cache = ObservationPrefixCache(architecture, compile_enabled=False)

    def broken_prefix(state):
        state.hidden_states = state.hidden_states * 100
        raise RuntimeError("compiler failure")

    def broken_joint(state, action, keys, values):
        state.hidden_states = state.hidden_states * 100
        action.payload.x_action = action.payload.x_action * 100
        raise RuntimeError("compiler failure")

    cache._compiled_prefill = broken_prefix
    cache._compiled_joint = broken_joint
    with torch.no_grad(), cache.generation():
        expected_v, expected_a = _dense(architecture, *_states(initial, 1))
        actual_v, actual_a = cache.run(*_states(initial, 1))
        torch.testing.assert_close(actual_v.hidden_states, expected_v, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual_a.payload.x_action, expected_a, rtol=1e-5, atol=1e-6)
        assert cache.stats["compile_failures"] == 2
        assert not cache.stats["compile_active"]
