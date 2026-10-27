"""Request-scoped cross-attention K/V precomputation for Wan dual-system serving."""

import threading
from contextlib import contextmanager

import torch
import torch.nn.functional as F


def compact_padding(context, context_mask):
    """Retain visible tokens, including non-contiguous proprio, in 32-token buckets."""
    if context.shape[0] != 1 or context_mask is None:
        raise ValueError("Compacted conditioning currently requires batch=1 and an explicit padding mask.")
    mask = context_mask
    if mask.ndim not in (2, 3, 4) or mask.shape[0] != context.shape[0]:
        raise ValueError("Compaction requires a batch-aligned 2D, 3D, or 4D mask.")
    if mask.dtype != torch.bool or mask.shape[-1] != context.shape[1]:
        raise ValueError("Compaction requires a boolean context padding mask.")
    while mask.ndim > 2:
        if mask.shape[1] > 1 and mask.stride(1) != 0:
            raise ValueError("Compaction requires a query/head-invariant padding mask.")
        mask = mask[:, 0]
    indices = mask[0].nonzero(as_tuple=False).flatten()
    valid = indices.numel()
    capacity = max(32, ((valid + 31) // 32) * 32)
    compact = context.new_zeros((1, capacity, context.shape[-1]))
    compact[:, :valid].copy_(context.index_select(1, indices))
    mask = torch.arange(capacity, device=context.device)[None, :] < valid
    return compact, mask


class _PackedKV:
    def __init__(self, modules, *, compact_context=False):
        self.modules = list(modules)
        self.compact_context = compact_context
        self.mask = None
        self.weight = self.bias = self.norm_weight = None
        self.keys = self.values = None
        self.signature = None
        self.rebuilds = 0

    def refresh_weights(self):
        parameters = [
            parameter
            for module in self.modules
            for parameter in (module.k.weight, module.k.bias, module.v.weight, module.v.bias, module.norm_k.weight)
            if parameter is not None
        ]
        signature = (
            tuple(
                (parameter._version, parameter.data_ptr(), parameter.dtype, parameter.device)
                for parameter in parameters
            ),
            tuple(module.norm_k.eps for module in self.modules),
        )
        if signature == self.signature:
            return
        widths = {module.k.out_features for module in self.modules}
        inputs = {module.k.in_features for module in self.modules}
        eps = {module.norm_k.eps for module in self.modules}
        if len(widths) != 1 or len(inputs) != 1 or len(eps) != 1:
            raise ValueError("Packed conditioning K/V requires matching projection dimensions and RMSNorm epsilon.")
        self.width = widths.pop()
        self.eps = eps.pop()
        weights, biases = [], []
        for module in self.modules:
            for projection in (module.k, module.v):
                weights.append(projection.weight.detach())
                biases.append(
                    projection.bias.detach()
                    if projection.bias is not None
                    else projection.weight.new_zeros(projection.out_features)
                )
        self.weight = torch.cat(weights, dim=0)
        self.bias = torch.cat(biases, dim=0)
        self.norm_weight = torch.stack([module.norm_k.weight.detach() for module in self.modules])
        self.signature = signature
        self.rebuilds += 1

    def prepare(self, context, context_mask=None):
        self.refresh_weights()
        if self.compact_context:
            context, mask = compact_padding(context, context_mask)
            mask = mask[:, None, None, :]
            if self.mask is None or self.mask.shape != mask.shape or self.mask.device != mask.device:
                self.mask = mask
            else:
                self.mask.copy_(mask)
        batch, length, _ = context.shape
        projected = F.linear(context, self.weight, self.bias)
        projected = projected.view(batch, length, len(self.modules), 2, self.width)
        keys = projected[:, :, :, 0, :].permute(2, 0, 1, 3).contiguous()
        values = projected[:, :, :, 1, :].permute(2, 0, 1, 3).contiguous()
        fp32 = keys.float()
        keys = (fp32 * torch.rsqrt(fp32.square().mean(dim=-1, keepdim=True) + self.eps)).to(keys.dtype)
        keys = keys * self.norm_weight[:, None, None, :]
        if (
            self.keys is None
            or self.keys.shape != keys.shape
            or self.keys.device != keys.device
            or self.keys.dtype != keys.dtype
        ):
            self.keys, self.values = keys, values
        else:
            # Stable storage addresses let the compiled graph see new conditions.
            self.keys.copy_(keys)
            self.values.copy_(values)
        for index, module in enumerate(self.modules):
            module._serving_key_cache = self.keys[index]
            module._serving_value_cache = self.values[index]
            module._serving_context_mask = self.mask
            module._serving_compact_context = self.compact_context
            module._serving_kv_active = True


class ConditionalKVCache:
    """Reuse only conditions invariant within one generation; never cache noisy states."""

    def __init__(self, architecture, *, compact_context=False, precompute_context=False):
        from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
        from openwam.model.video_backbone.wan_backbone import WanBase

        if not isinstance(architecture, DualSystemSelfAttnArchitecture) or not isinstance(
            architecture.video_backbone, WanBase
        ):
            raise ValueError("Conditional K/V precomputation currently supports Wan dual-system joint self-attention.")
        video = [block.cross_attn for block in architecture.video_backbone.dit.blocks]
        action = [block.cross_attn for block in architecture.action_backbone.blocks]
        if any(module.has_image_input for module in video):
            raise ValueError("Separate CLIP image cross-attention is not supported by this conditioning cache.")
        self.video, self.action = (
            _PackedKV(video, compact_context=compact_context),
            _PackedKV(action, compact_context=compact_context),
        )
        self.modules = video + action
        for module in self.modules:
            if "_serving_key_cache" not in module._buffers:
                module.register_buffer("_serving_key_cache", None, persistent=False)
                module.register_buffer("_serving_value_cache", None, persistent=False)
            module._serving_kv_active = False
            if "_serving_context_mask" not in module._buffers:
                module.register_buffer("_serving_context_mask", None, persistent=False)
            module._serving_compact_context = False
        self.architecture = architecture
        self.compact_context = compact_context
        self.precompute_context = precompute_context
        self.context_inputs = None
        self.embeddings = {}
        self.embedding_projections = 0
        architecture.video_backbone._serving_conditioning_cache = self
        architecture.action_backbone._serving_conditioning_cache = self
        self.ready = False
        self.active = False
        self.lock = threading.RLock()
        self.generations = self.projections = 0

    @contextmanager
    def generation(self):
        with self.lock:
            if self.architecture.training or torch.is_grad_enabled():
                raise RuntimeError("Conditional K/V reuse requires eval and no-grad inference.")
            self.ready = False
            self.context_inputs = None
            self.embeddings.clear()
            self.active = True
            self.generations += 1
            try:
                yield
            finally:
                for module in self.modules:
                    module._serving_kv_active = False
                self.ready = False
                self.context_inputs = None
                self.embeddings.clear()
                self.active = False

    def prepare_inputs(self, inputs):
        if self.context_inputs is None:
            context = inputs["context"]
            mask = inputs.get("context_mask")
            if mask is None:
                seq_lens = inputs.get("seq_lens")
                positions = torch.arange(context.shape[1], device=context.device)[None, :]
                mask = (
                    positions < seq_lens.to(context.device)[:, None]
                    if seq_lens is not None
                    else torch.ones(context.shape[:2], dtype=torch.bool, device=context.device)
                )
            if self.compact_context:
                context, mask = compact_padding(context, mask)
            self.context_inputs = context, mask
        result = dict(inputs)
        result["context"], result["context_mask"] = self.context_inputs
        result.pop("seq_lens", None)
        return result

    def embed_context(self, stream, embedding, context):
        if stream not in self.embeddings:
            self.embeddings[stream] = embedding(context)
            self.embedding_projections += 1
        return self.embeddings[stream]

    def prepare(self, video_context, action_context, video_mask=None, action_mask=None):
        if self.ready:
            return
        if video_context is None or action_context is None:
            raise ValueError("Both streams require conditioning context.")
        self.video.prepare(video_context, video_mask)
        self.action.prepare(action_context, action_mask)
        self.ready = True
        self.projections += 1
