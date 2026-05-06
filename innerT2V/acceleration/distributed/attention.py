import torch
from yunchang.ring.utils import RingComm, update_out_and_lse

from utils.logger import get_logger

logger = get_logger()

try:
    from flash_attn_interface import flash_attn_func
    FA3_AVAILABLE = True
except Exception as e:
    logger.warning(f"fa3 is not available with error: {e}")
    flash_attn_func = None
    FA3_AVAILABLE = False


if not FA3_AVAILABLE:
    ring_flash_attn_func = None
else:
    from flash_attn_interface import (
        _flash_attn_forward,
        _flash_attn_backward,
    )

    def ring_flash_attn_forward(
        process_group,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale,
        causal=True,
        window_size=(-1, -1),
        softcap=0.0,
        deterministic=False,
        sm_margin=0,
    ):
        comm = RingComm(process_group)

        out = None
        lse = None

        next_k, next_v = None, None

        for step in range(comm.world_size):
            if step + 1 != comm.world_size:
                next_k: torch.Tensor = comm.send_recv(k)
                next_v: torch.Tensor = comm.send_recv(v)
                comm.commit()

            if not causal or step <= comm.rank:
                block_out, block_lse, *rest = _flash_attn_forward(
                    q,
                    k,
                    v,
                    None, None,
                    None,
                    None,
                    None, None, None,
                    None, None,
                    None, None,
                    None, None, None,
                    None, None,
                    None, None, None,
                    softmax_scale,
                    causal=causal and step == 0,
                    window_size=window_size,
                    softcap=softcap,
                    num_splits=1,
                    pack_gqa=None,
                    sm_margin=sm_margin,
                )
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)

            if step + 1 != comm.world_size:
                comm.wait()
                k = next_k
                v = next_v

        out = out.to(q.dtype)
        lse = lse.squeeze(dim=-1).transpose(1, 2)
        return out, lse


    def ring_flash_attn_backward(
        process_group,
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        softmax_scale,
        causal=True,
        window_size=(-1, -1),
        softcap=0.0,
        deterministic=False,
        sm_margin=0,
    ):
        kv_comm = RingComm(process_group)
        d_kv_comm = RingComm(process_group)
        dq, dk, dv = None, None, None
        next_dk, next_dv = None, None

        block_dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        block_dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
        block_dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

        next_dk, next_dv = None, None
        next_k, next_v = None, None

        for step in range(kv_comm.world_size):
            if step + 1 != kv_comm.world_size:
                next_k = kv_comm.send_recv(k)
                next_v = kv_comm.send_recv(v)
                kv_comm.commit()
            if step <= kv_comm.rank or not causal:
                _flash_attn_backward(
                    dout,
                    q,
                    k,
                    v,
                    out,
                    softmax_lse,
                    None, None,
                    None, None,
                    None, None,
                    block_dq_buffer,
                    block_dk_buffer,
                    block_dv_buffer,
                    softmax_scale,
                    causal and step == 0,
                    window_size,
                    softcap,
                    deterministic,
                    sm_margin,
                )

                if dq is None:
                    dq = block_dq_buffer.to(torch.float32)
                    dk = block_dk_buffer.to(torch.float32)
                    dv = block_dv_buffer.to(torch.float32)
                else:
                    dq += block_dq_buffer
                    d_kv_comm.wait()
                    dk = block_dk_buffer + next_dk
                    dv = block_dv_buffer + next_dv
            elif step != 0:
                d_kv_comm.wait()
                dk = next_dk
                dv = next_dv

            if step + 1 != kv_comm.world_size:
                kv_comm.wait()
                k = next_k
                v = next_v

            next_dk = d_kv_comm.send_recv(dk)
            next_dv = d_kv_comm.send_recv(dv)
            d_kv_comm.commit()

        d_kv_comm.wait()

        return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)

    from ..sac_utils import sac_storage

    class RingFlashAttnFunc(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            q,
            k,
            v,
            softmax_scale,
            causal,
            window_size,
            softcap,
            return_softmax,
            deterministic,
            group,
            sac_id=None,
        ):
            if softmax_scale is None:
                softmax_scale = q.shape[-1] ** (-0.5)

            if sac_id is not None and sac_id in sac_storage:
                out, softmax_lse = sac_storage.pop(sac_id)
                ctx.save_for_backward(q, k, v, out, softmax_lse)
            else:
                k = k.contiguous()
                v = v.contiguous()
                out, softmax_lse = ring_flash_attn_forward(
                    group,
                    q,
                    k,
                    v,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    softcap=softcap,
                    deterministic=deterministic,
                )
                ctx.save_for_backward(q, k, v, out, softmax_lse)
                sac_storage[sac_id] = (out, softmax_lse)
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.deterministic = deterministic
            ctx.group = group
            return out if not return_softmax else (out, softmax_lse, None)

        @staticmethod
        def backward(ctx, dout, *args):
            q, k, v, out, softmax_lse = ctx.saved_tensors
            dq, dk, dv = ring_flash_attn_backward(
                ctx.group,
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                softmax_scale=ctx.softmax_scale,
                causal=ctx.causal,
                window_size=ctx.window_size,
                softcap=ctx.softcap,
                deterministic=ctx.deterministic,
            )
            return dq, dk, dv, None, None, None, None, None, None, None, None


    def ring_flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        group=None,
        attn_type=None,
        sac_id=None,
    ):
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
        return RingFlashAttnFunc.apply(
            q,
            k,
            v,
            softmax_scale,
            causal,
            window_size,
            softcap,
            return_attn_probs,
            deterministic,
            group,
            sac_id,
        )
