import torch
import functools
from typing import Dict, Any
from diffusers.utils.import_utils import is_torch_version


if is_torch_version(">=", "2.5.0"):
    from torch.utils.checkpoint import create_selective_checkpoint_contexts
    attn_ops = [
        torch.ops.aten._scaled_dot_product_flash_attention,
        torch.ops.aten._scaled_dot_product_efficient_attention,
        torch.ops.aten._flash_attention_forward,
        torch.ops.aten._efficient_attention_forward,
    ]
    def policy_fn(ctx, op, *args, **kwargs):
        if op in attn_ops:
            return torch.utils.checkpoint.CheckpointPolicy.MUST_SAVE
        else:
            return torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
else:
    from torch.utils.checkpoint import _pt2_selective_checkpoint_context_fn_gen
    attn_ops = [
        torch.ops.aten._scaled_dot_product_flash_attention.default,
        torch.ops.aten._scaled_dot_product_efficient_attention.default,
        torch.ops.aten._flash_attention_forward.default,
        torch.ops.aten._efficient_attention_forward.default,
    ]
    def policy_fn(ctx, op, *args, **kwargs):
        return op in attn_ops
    def create_selective_checkpoint_contexts(policy_fn, *args, **kwargs):
        return _pt2_selective_checkpoint_context_fn_gen(policy_fn)

old_torch_checkpoint_fn = torch.utils.checkpoint.checkpoint

def new_torch_checkpoint_fn(function, *args, **kwargs):
    ckpt_kwargs: Dict[str, Any] = {
        "use_reentrant": False,
        "context_fn": functools.partial(create_selective_checkpoint_contexts, policy_fn),
    } if is_torch_version(">=", "1.11.0") else {}
    kwargs.update(ckpt_kwargs)
    return old_torch_checkpoint_fn(function, *args, **kwargs)


def optimize_activation_checkpoint():
    from .sac_utils import sac_storage
    sac_storage.enable()
    torch.utils.checkpoint.checkpoint = new_torch_checkpoint_fn
