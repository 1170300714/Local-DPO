import uuid
import torch
import inspect
import functools

from transformer.cogvideox_improved_transformer_3d import CogVideoXUnifyAttnProcessor2_0
from utils.logger import get_logger

logger = get_logger()


try:
    from sageattention import sageattn
    q = k = v = torch.randn(2, 48, 16, 64).to('cuda', dtype=torch.bfloat16)
    sageattn(q, k, v, is_causal=False)
    def sage_attn_wrapper(
        q,
        k,
        v,
        sac_id=None,
        **kwargs,
    ):
        return sageattn(q, k, v, **kwargs)
    SAGE_AVAIABLE = True
except Exception as e:
    logger.warning(f"sageattention is not available with error: {e}")
    sage_attn_wrapper = None
    SAGE_AVAIABLE = False

try:
    from flash_attn_interface import flash_attn_func
    FA3_AVAILABLE = True
except Exception as e:
    logger.warning(f"fa3 is not available with error: {e}")
    flash_attn_func = None
    FA3_AVAILABLE = False


if not FA3_AVAILABLE:
    flash_attn_v3_wrapper = None
else:
    from flash_attn_interface import FlashAttnFunc, _flash_attn_forward

    from .sac_utils import sac_storage

    class FlashAttnSACFunc(FlashAttnFunc):

        @staticmethod
        def forward(
            ctx,
            q,
            k,
            v,
            softmax_scale,
            causal,
            qv=None,
            q_descale=None, k_descale=None, v_descale=None,
            window_size=(-1, -1),
            softcap=0.0,
            num_splits=1,
            pack_gqa=None,
            deterministic=False,
            sm_margin=0,
            sac_id=None,
        ):
            if softmax_scale is None:
                softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (-0.5)
            if sac_id is not None and sac_id in sac_storage:
                out, softmax_lse = sac_storage.pop(sac_id)
                ctx.save_for_backward(q, k, v, out, softmax_lse)
            else:
                out, softmax_lse, *rest = _flash_attn_forward(
                    q,
                    k,
                    v,
                    None, None,
                    qv,
                    None,
                    None, None, None,
                    None, None,
                    None, None,
                    None, None, None,
                    None, None,
                    q_descale, k_descale, v_descale,
                    softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    softcap=softcap,
                    num_splits=num_splits,
                    pack_gqa=pack_gqa,
                    sm_margin=sm_margin,
                )
                ctx.save_for_backward(q, k, v, out, softmax_lse)
                sac_storage[sac_id] = (out, softmax_lse)
            ctx.softmax_scale = softmax_scale
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.causal = causal
            ctx.deterministic = deterministic
            ctx.sm_margin = sm_margin
            return out, softmax_lse

    def flash_attn_sac_func(
        q,
        k,
        v,
        softmax_scale=None,
        causal=False,
        qv=None,
        q_descale=None, k_descale=None, v_descale=None,
        window_size=(-1, -1),
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
        deterministic=False,
        sm_margin=0,
        sac_id=None,
    ):
        return FlashAttnSACFunc.apply(
            q,
            k,
            v,
            softmax_scale,
            causal,
            qv,
            q_descale, k_descale, v_descale,
            window_size,
            softcap,
            num_splits,
            pack_gqa,
            deterministic,
            sm_margin,
            sac_id,
        )

    def flash_attn_v3_wrapper(
        q,
        k,
        v,
        attention_mask=None,
        dropout_p=0.0,
        is_causal=False,
        sac_id=None,
        **kwargs,
    ):
        assert attention_mask is None
        assert dropout_p == 0.0
        hidden_states, *_ = flash_attn_sac_func(
            q.transpose(1, 2).to(dtype=torch.bfloat16),
            k.transpose(1, 2).to(dtype=torch.bfloat16),
            v.transpose(1, 2).to(dtype=torch.bfloat16),
            causal=is_causal,
            sac_id=sac_id,
        )
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


def optimize_transformer(transformer, attn_type='fa3'):
    if 'ppu' in torch.cuda.get_device_name().lower():
        logger.warning("[Perf] Disable attention optimization on PPU cluster")
        return transformer

    if attn_type == 'fa3' and not FA3_AVAILABLE:
        logger.warning("fa3 is not available, skip optimizing")
        return transformer
    if attn_type == 'sage' and not SAGE_AVAIABLE:
        logger.warning("sageattention is not available, skip optimizing")
        return transformer
    attn_fn = {
        'fa3': flash_attn_v3_wrapper,
        'sage': sage_attn_wrapper
    }[attn_type]

    for i, block in enumerate(transformer.transformer_blocks):
        if not hasattr(block.attn1.processor, 'attn_func'):
            block.attn1.processor = CogVideoXUnifyAttnProcessor2_0()

        if 'sac_id' in inspect.signature(attn_fn).parameters:
            block.attn1.processor.attn_func = functools.partial(attn_fn, sac_id=str(uuid.uuid4()))
        else:
            block.attn1.processor.attn_func = attn_fn

        if hasattr(block, 'attn_cross'):
            if 'sac_id' in inspect.signature(attn_fn).parameters:
                block.attn_cross.processor.attn_func = functools.partial(attn_fn, sac_id=str(uuid.uuid4()))
            else:
                block.attn_cross.processor.attn_func = attn_fn
    return transformer
