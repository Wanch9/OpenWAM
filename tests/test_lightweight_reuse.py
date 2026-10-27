from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from openwam.deploy.optimizations.lightweight_reuse import LightweightResidualReuse
from openwam.deploy.optimizations.residual_cache import TransformerResidualCache
from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from openwam.model.video_backbone.wan import dit_forward
from openwam.model.video_backbone.wan.models.dit import WanModel
from openwam.model.video_backbone.wan_backbone import Wan22Ti2v


class _Driver:
    def __init__(self):
        self.calls = 0

    def run_joint_loop(self, video, action, **kwargs):
        self.calls += 1
        # Couple both modalities at refreshes; the optimization must preserve
        # their already-defined joint residual, not replace the action head.
        video_signal = video.hidden_states.mean()
        action_signal = action.payload.x_action.mean()
        video.hidden_states = video.hidden_states + video.time_mod[:, :, 0] + action_signal
        action.payload.x_action = action.payload.x_action + action.payload.t_mod[:, :1] + video_signal
        return video, action


def _setup(dtype=torch.float32, steps=5):
    torch.manual_seed(719)
    dit = WanModel(
        dim=24, in_dim=2, ffn_dim=48, out_dim=2, text_dim=8, freq_dim=8,
        eps=1e-6, patch_size=(1, 2, 2), num_heads=2, num_layers=1,
        has_image_input=False, seperated_timestep=True,
        require_vae_embedding=False, require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,
    )
    video = Wan22Ti2v(SimpleNamespace(dit=dit), text_dim=8).to(dtype=dtype).eval()
    action = ActionDiT(
        action_dim=4, dim=12, ffn_dim=24, num_heads=2, num_layers=1,
        video_dim=24, bridge_layers=(0,), variant="joint_self_attn",
        attn_head_dim=12, text_dim=8, freq_dim=8,
    ).to(dtype=dtype).eval()
    cache = TransformerResidualCache((0, 3) if steps > 3 else (0,), total_steps=steps)
    driver = _Driver()
    arch = SimpleNamespace(
        video_backbone=video, action_backbone=action, training=False,
        _mot_driver=driver, _serving_residual_cache=cache,
        _append_proprio_context_token=lambda inputs, proprio: inputs,
        _compiled_mot_run_joint_loop=None,
    )
    counts = {key: 0 for key in ("video_mod", "action_time", "video_head", "action_head")}
    for key, module in (
        ("video_mod", dit.time_projection), ("action_time", action.time_embedding),
        ("video_head", dit.head), ("action_head", action.action_decoder),
    ):
        def increment(module, args, output, key=key):
            counts[key] += 1
        module.register_forward_hook(increment)
    return arch, counts


def _inputs(step, dtype=torch.float32, fuse=True):
    generator = torch.Generator().manual_seed(800 + step)
    return (
        torch.randn(1, 3, 4, generator=generator, dtype=dtype),
        torch.tensor([700.0 - step * 100.0], dtype=dtype),
        {
            "latents": torch.randn(1, 2, 3, 4, 4, generator=generator, dtype=dtype),
            "timestep": torch.tensor([700.0 - step * 100.0], dtype=dtype),
            "context": torch.randn(1, 3, 8, generator=generator, dtype=dtype),
            "context_mask": torch.tensor([[True, True, False]]),
            "fuse_vae_embedding_in_latents": fuse,
            "num_clean_prefix_frames": 1,
        },
    )


def _run(lightweight, dtype=torch.float32, fuse=True):
    arch, counts = _setup(dtype)
    runner = LightweightResidualReuse(arch)
    arch._serving_lightweight_reuse = runner if lightweight else None
    output = []
    with torch.no_grad(), arch._serving_residual_cache.generation(), runner.generation() if lightweight else nullcontext():
        for step in range(5):
            actions, timestep, inputs = _inputs(step, dtype, fuse)
            result = DualSystemSelfAttnArchitecture.forward(arch, actions, timestep, **inputs)
            output.append(tuple(value.clone() for value in result))
    return output, arch, counts, runner


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_actual_wan_action_heads_are_bitwise_equal_with_current_inputs(dtype):
    reference, _, reference_counts, _ = _run(False, dtype)
    actual, arch, counts, runner = _run(True, dtype)
    for expected_pair, actual_pair in zip(reference, actual):
        for expected, value in zip(expected_pair, actual_pair):
            assert torch.equal(value, expected)
    assert reference_counts == {"video_mod": 5, "action_time": 5, "video_head": 5, "action_head": 5}
    assert counts == {"video_mod": 2, "action_time": 2, "video_head": 5, "action_head": 5}
    assert arch._mot_driver.calls == 2
    assert arch._serving_residual_cache.stats["head_evals"] == 5
    assert runner.stats == {"active": False, "reuse_evals": 3, "fallbacks": 0}


def test_unsupported_time_modulation_branch_falls_back_to_original_forward():
    reference, _, _, _ = _run(False, fuse=False)
    actual, _, counts, runner = _run(True, fuse=False)
    for expected_pair, actual_pair in zip(reference, actual):
        assert all(torch.equal(value, expected) for value, expected in zip(actual_pair, expected_pair))
    assert counts["video_mod"] == counts["action_time"] == 5
    assert runner.stats["fallbacks"] == 3 and runner.stats["reuse_evals"] == 0


def test_time_cache_integration_preserves_complete_original_compute_callback():
    arch, counts = _setup()
    runner = LightweightResidualReuse(arch)
    _, _, inputs = _inputs(1)
    video = arch.video_backbone
    expected, _ = dit_forward.build_time_modulation(
        video.dit, inputs["timestep"], inputs["latents"],
        patch_size=video._dit_patch_size, fuse_vae_embedding_in_latents=True,
        force_per_token_t_mod=True, num_clean_prefix_frames=1,
        zero_clean_prefix_t_mod=True, has_first_frame_latents=False,
    )

    class TimeCache:
        active = True
        stored = None

        def video(self, dit, timestep, latents, *, compute, **kwargs):
            assert kwargs["force_per_token_t_mod"] and kwargs["zero_clean_prefix_t_mod"]
            assert kwargs["patch_size"] == (1, 2, 2)
            if self.stored is None:
                self.stored = compute()
                assert len(self.stored) == 2
            return self.stored

    video._serving_time_cache = TimeCache()
    with torch.no_grad():
        actual = runner._time_embed(video, inputs["timestep"], inputs["latents"], inputs)
        warmed = runner._time_embed(video, inputs["timestep"], inputs["latents"], inputs)
    assert torch.equal(actual, expected) and warmed is actual
    assert counts["video_mod"] == 2  # one reference, one cold full callback; no warm projection


def test_generation_cleanup_reentrancy_and_no_grad_guard():
    arch, _ = _setup()
    runner = LightweightResidualReuse(arch)
    actions, timestep, inputs = _inputs(1)
    with pytest.raises(RuntimeError, match="active generation"):
        runner.run(actions, timestep, **inputs)
    with runner.generation():
        with pytest.raises(RuntimeError, match="concurrent/reentrant"):
            with runner.generation():
                pass
        assert runner.active
        with pytest.raises(RuntimeError, match="no-grad"):
            runner.run(actions, timestep, **inputs)
    assert not runner.active
    with pytest.raises(RuntimeError, match="injected"):
        with runner.generation():
            raise RuntimeError("injected")
    assert not runner.active
    with runner.generation():
        assert runner.stats["reuse_evals"] == 0


def test_refresh_dispatch_is_rejected_without_advancing_existing_cache():
    arch, _ = _setup(steps=1)
    runner = LightweightResidualReuse(arch)
    actions, timestep, inputs = _inputs(0)
    with torch.no_grad(), arch._serving_residual_cache.generation(), runner.generation():
        with pytest.raises(RuntimeError, match="non-refresh"):
            runner.run(actions, timestep, **inputs)
        assert arch._serving_residual_cache.stats["steps_completed"] == 0
        DualSystemSelfAttnArchitecture.forward(arch, actions, timestep, **inputs)


def test_existing_residual_shape_validation_is_retained_on_lightweight_steps():
    arch, _ = _setup(steps=2)
    runner = LightweightResidualReuse(arch)
    arch._serving_lightweight_reuse = runner
    with torch.no_grad(), pytest.raises(ValueError, match="changed within one generation"):
        with arch._serving_residual_cache.generation(), runner.generation():
            actions, timestep, inputs = _inputs(0)
            DualSystemSelfAttnArchitecture.forward(arch, actions, timestep, **inputs)
            runner.run(actions[:, :2], timestep, **inputs)
    assert not runner.active and not arch._serving_residual_cache.active


def _enable_proprio(arch):
    arch._use_proprioception_context = True
    arch.proprio_dim = 4
    arch.proprio_encoder = torch.nn.Linear(4, 8).eval()
    arch._append_proprio_context_token = lambda inputs, proprio: BaseWAMArchitecture._append_proprio_context_token(
        arch, inputs, proprio
    )


@pytest.mark.parametrize(
    "malformed",
    ["context_width", "context_mask_rank", "context_mask_length", "seq_lens_shape",
     "missing_proprio", "proprio_width", "proprio_mask_shape"],
)
def test_invalid_conditioning_metadata_falls_back_and_preserves_original_error(malformed):
    def run(lightweight):
        arch, _ = _setup(steps=2)
        use_proprio = "proprio" in malformed
        if use_proprio:
            _enable_proprio(arch)
        runner = LightweightResidualReuse(arch)
        arch._serving_lightweight_reuse = runner if lightweight else None
        try:
            with torch.no_grad(), arch._serving_residual_cache.generation(), runner.generation():
                actions, timestep, inputs = _inputs(0)
                proprio = torch.ones(4) if use_proprio else None
                DualSystemSelfAttnArchitecture.forward(arch, actions, timestep, proprio=proprio, **inputs)
                actions, timestep, inputs = _inputs(1)
                if malformed == "context_width":
                    inputs["context"] = torch.ones(1, 3, 9)
                elif malformed == "context_mask_rank":
                    inputs["context_mask"] = torch.ones(1, 1, 3, dtype=torch.bool)
                elif malformed == "context_mask_length":
                    inputs["context_mask"] = torch.ones(1, 2, dtype=torch.bool)
                elif malformed == "seq_lens_shape":
                    inputs.pop("context_mask")
                    inputs["seq_lens"] = torch.tensor([[3]])
                elif malformed == "missing_proprio":
                    proprio = None
                elif malformed == "proprio_width":
                    proprio = torch.ones(5)
                elif malformed == "proprio_mask_shape":
                    inputs["_proprio_sample_mask"] = torch.ones(1, 2, dtype=torch.bool)
                DualSystemSelfAttnArchitecture.forward(arch, actions, timestep, proprio=proprio, **inputs)
        except (ValueError, RuntimeError, AttributeError) as exc:
            assert not runner.active
            return type(exc), str(exc), runner.stats
        pytest.fail("Malformed input unexpectedly bypassed the original validation")

    expected_type, expected_message, _ = run(False)
    actual_type, actual_message, stats = run(True)
    assert actual_type is expected_type and actual_message == expected_message
    assert stats["fallbacks"] == 1 and stats["reuse_evals"] == 0


def test_valid_proprio_metadata_does_not_repeat_linear_projection_on_reuse():
    def run(lightweight):
        arch, _ = _setup()
        _enable_proprio(arch)
        calls = []
        arch.proprio_encoder.register_forward_hook(lambda *args: calls.append(1))
        runner = LightweightResidualReuse(arch)
        arch._serving_lightweight_reuse = runner if lightweight else None
        outputs = []
        with torch.no_grad(), arch._serving_residual_cache.generation(), runner.generation():
            for step in range(5):
                actions, timestep, inputs = _inputs(step)
                result = DualSystemSelfAttnArchitecture.forward(
                    arch, actions, timestep, proprio=torch.ones(4), **inputs
                )
                outputs.append(tuple(value.clone() for value in result))
        return outputs, len(calls)

    reference, baseline_calls = run(False)
    actual, lightweight_calls = run(True)
    for expected_pair, actual_pair in zip(reference, actual):
        assert all(torch.equal(value, expected) for value, expected in zip(actual_pair, expected_pair))
    assert baseline_calls == 5 and lightweight_calls == 2
