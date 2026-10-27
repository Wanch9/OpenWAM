from types import SimpleNamespace

import pytest
import torch

from openwam.deploy.optimizations.residual_cache import TransformerResidualCache


@pytest.mark.parametrize("refresh_steps", [(0, 3), (0, 3, 6, 9), tuple(range(10))])
def test_fixed_budget_and_current_input_reuse(refresh_steps):
    cache = TransformerResidualCache(refresh_steps)
    calls = []
    with torch.no_grad(), cache.generation():
        for step in range(10):
            v = torch.full((1, 3, 4), float(step))
            a = torch.full((1, 2, 4), float(step * 2))

            def full():
                calls.append(step)
                return v + step + 1, a + 2 * step + 1

            actual_v, actual_a = cache.run(v, a, full)
            last_refresh = max(x for x in refresh_steps if x <= step)
            torch.testing.assert_close(actual_v, v + last_refresh + 1)
            torch.testing.assert_close(actual_a, a + 2 * last_refresh + 1)
            cache.record_head_evaluation()
    assert calls == list(refresh_steps)
    assert cache.stats["full_evals"] == len(refresh_steps)
    assert cache.stats["reused_evals"] == 10 - len(refresh_steps)
    assert cache.stats["head_evals"] == cache.stats["steps_completed"] == 10
    assert not cache.active
    assert cache._video_residual is None and cache._action_residual is None


def test_independent_fp32_residual_and_refresh_output_identity():
    cache = TransformerResidualCache((0,), total_steps=2)
    v = torch.tensor([[[1.0, 1.0078125]]], dtype=torch.bfloat16)
    a = v.clone()
    with torch.no_grad(), cache.generation():
        out_v, out_a = v + 2, a - 3
        actual = cache.run(v, a, lambda: (out_v, out_a))
        assert actual[0] is out_v and actual[1] is out_a
        assert cache._video_residual.dtype == cache._action_residual.dtype == torch.float32
        expected_v = cache._video_residual.clone()
        expected_a = cache._action_residual.clone()
        cache.record_head_evaluation()
        # Simulate input mutation and CUDA Graph reusing its output allocations.
        v.fill_(100)
        a.fill_(100)
        out_v.fill_(999)
        out_a.fill_(999)
        next_v, next_a = cache.run(v, a, lambda: pytest.fail("unexpected full forward"))
        torch.testing.assert_close(next_v, (v.float() + expected_v).to(v.dtype))
        torch.testing.assert_close(next_a, (a.float() + expected_a).to(a.dtype))
        cache.record_head_evaluation()


def test_input_snapshot_survives_inplace_transformer():
    cache = TransformerResidualCache((0,), total_steps=2)
    v, a = torch.ones(1, 2, 3), torch.ones(1, 2, 3)
    with torch.no_grad(), cache.generation():
        cache.run(v, a, lambda: (v.add_(7), a.add_(11)))
        cache.record_head_evaluation()
        actual_v, actual_a = cache.run(torch.zeros_like(v), torch.zeros_like(a), lambda: None)
        torch.testing.assert_close(actual_v, torch.full_like(v, 7))
        torch.testing.assert_close(actual_a, torch.full_like(a, 11))
        cache.record_head_evaluation()


@pytest.mark.parametrize("steps", [(), (1,), (0, 0), (0, 10), (0, -1), (0, True), (0, 1.0)])
def test_invalid_refresh_schedule_rejected(steps):
    with pytest.raises(ValueError):
        TransformerResidualCache(steps)


def test_request_reset_and_exception_cleanup():
    cache = TransformerResidualCache((0,), total_steps=2)
    x = torch.zeros(1, 2, 3)
    with torch.no_grad(), pytest.raises(RuntimeError, match="request failed"):
        with cache.generation():
            cache.run(x, x, lambda: (x + 1, x + 2))
            cache.record_head_evaluation()
            raise RuntimeError("request failed")
    assert not cache.active and cache._video_residual is None
    with torch.no_grad(), cache.generation():
        cache.run(x, x, lambda: (x + 5, x + 9))
        cache.record_head_evaluation()
        v, a = cache.run(x, x, lambda: None)
        torch.testing.assert_close(v, x + 5)
        torch.testing.assert_close(a, x + 9)
        cache.record_head_evaluation()
    assert cache.stats["full_evals"] == 1


def test_reentrant_generation_rejected_without_corrupting_outer():
    cache = TransformerResidualCache((0,), total_steps=1)
    with torch.no_grad(), cache.generation():
        with pytest.raises(RuntimeError, match="concurrent/reentrant"):
            with cache.generation():
                pass
        x = torch.ones(1)
        cache.run(x, x, lambda: (x, x))
        cache.record_head_evaluation()


def test_missing_heads_or_updates_rejected_and_cleaned():
    cache = TransformerResidualCache((0,), total_steps=1)
    x = torch.ones(1)
    with torch.no_grad(), pytest.raises(RuntimeError, match="Incomplete"):
        with cache.generation():
            cache.run(x, x, lambda: (x, x))
    assert not cache.active and cache._video_residual is None
    with pytest.raises(RuntimeError, match="Incomplete"):
        with cache.generation():
            pass


def test_gradients_and_changed_shape_rejected():
    cache = TransformerResidualCache((0,), total_steps=2)
    x = torch.ones(1)
    with pytest.raises(RuntimeError, match="no-grad"):
        with cache.generation():
            cache.run(x, x, lambda: (x, x))
    with torch.no_grad(), pytest.raises(ValueError, match="changed"):
        with cache.generation():
            cache.run(x, x, lambda: (x, x))
            cache.record_head_evaluation()
            cache.run(torch.ones(2), x, lambda: None)
    assert not cache.active and cache._video_residual is None


class _VideoMock:
    def __init__(self):
        self.prepares = self.heads = 0

    def prepare(self, **kwargs):
        self.prepares += 1
        return SimpleNamespace(
            hidden_states=kwargs["latents"] * 2 + kwargs["timestep"],
            time=kwargs["timestep"],
            context=kwargs["context"],
            context_mask=None,
        )

    def finalize(self, state):
        self.heads += 1
        return state.hidden_states + 5 * state.time


class _ActionMock:
    def __init__(self):
        self.prepares = self.heads = 0

    def prepare_state(self, actions, timestep, **kwargs):
        self.prepares += 1
        return SimpleNamespace(
            payload=SimpleNamespace(
                x_action=actions * 3 + timestep,
                time=timestep,
                context=kwargs["context"],
                context_mask=None,
            )
        )

    def extract_prediction(self, state):
        self.heads += 1
        return state.payload.x_action + 4 * state.payload.time


class _DriverMock:
    def __init__(self):
        self.calls = 0

    def run_joint_loop(self, vstate, astate, **kwargs):
        self.calls += 1
        vstate.hidden_states = vstate.hidden_states + 2 * vstate.time + 7
        astate.payload.x_action = astate.payload.x_action + 3 * astate.payload.time + 11
        return vstate, astate


@pytest.mark.parametrize("compiled", [False, True])
def test_architecture_keeps_current_embeddings_and_heads_on_every_update(compiled):
    from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture

    cache = TransformerResidualCache((0, 3, 6, 9))
    vb, ab, driver = _VideoMock(), _ActionMock(), _DriverMock()
    arch = SimpleNamespace(
        video_backbone=vb,
        action_backbone=ab,
        training=False,
        _mot_driver=driver,
        _serving_residual_cache=cache,
        _append_proprio_context_token=lambda inputs, proprio: inputs,
        _compiled_mot_run_joint_loop=driver.run_joint_loop if compiled else None,
    )
    with torch.no_grad(), cache.generation():
        for step in range(10):
            image = torch.full((1, 3, 4), float(step * 2))
            action = torch.full((1, 2, 4), float(step * 3))
            timestep = torch.tensor([float(step)])
            video_out, action_out = DualSystemSelfAttnArchitecture.forward(
                arch, action, timestep, latents=image, timestep=timestep, context=torch.ones(1, 1, 4)
            )
            refreshed = max(s for s in cache.refresh_steps if s <= step)
            torch.testing.assert_close(video_out, image * 2 + 6 * step + 2 * refreshed + 7)
            torch.testing.assert_close(action_out, action * 3 + 5 * step + 3 * refreshed + 11)
    assert driver.calls == 4
    assert vb.prepares == ab.prepares == vb.heads == ab.heads == cache.stats["head_evals"] == 10
