import uuid
import inspect
import functools
from typing import Optional, Tuple

import torch
from yunchang import LongContextAttention

from extensions.xfuser.core.distributed import (
    get_sp_group,
    get_cfg_group,
)
from .attention import ring_flash_attn_func
from .collective import (
    split_forward_gather_backward,
    gather_forward_split_backward,
)
from transformer.cogvideox_improved_transformer_3d import (
    CogVideoXUnifyAttnProcessor2_0
)

from utils.logger import get_logger

logger = get_logger()


class CompatibleLongContextAttention(LongContextAttention):

    def __init__(self, sac_id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ring_attn_fn = functools.partial(ring_flash_attn_func, sac_id=sac_id)

    def forward(self, 
        q,
        k,
        v,
        attention_mask=None,
        dropout_p=0.0,
        is_causal=False,
        *args, **kwargs
    ):
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        assert attention_mask is None
        o = super().forward(q, k, v, dropout_p=dropout_p, causal=is_causal)
        return o.transpose(1, 2).contiguous()


def parallelize_transformer(transformer):

    for block in transformer.transformer_blocks:
        sac_id = str(uuid.uuid4())
        block.attn1.processor = CogVideoXUnifyAttnProcessor2_0(
            attn_func=CompatibleLongContextAttention(sac_id=sac_id))

    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: torch.LongTensor = None,
        pooled_projections: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *args,
        **kwargs,
    ):
        assert encoder_hidden_states.shape[-2] % get_sp_group().world_size == 0

        if self.config.patch_size_t is None:
            temporal_size = hidden_states.shape[1]
        else:
            temporal_size = hidden_states.shape[1] // self.config.patch_size_t

        hidden_states = split_forward_gather_backward(
            hidden_states, dim=-2, process_group=get_sp_group()
        )
        encoder_hidden_states = split_forward_gather_backward(
            encoder_hidden_states, dim=-2, process_group=get_sp_group()
        )

        if get_cfg_group().world_size > 1:
            if isinstance(timestep, torch.Tensor) and timestep.ndim != 0 and timestep.shape[0] == hidden_states.shape[0]:
                timestep = split_forward_gather_backward(
                    timestep, dim=0, process_group=get_cfg_group()
                )
            hidden_states = split_forward_gather_backward(
                hidden_states, dim=0, process_group=get_cfg_group()
            )
            encoder_hidden_states = split_forward_gather_backward(
                encoder_hidden_states, dim=0, process_group=get_cfg_group()
            )
            if pooled_projections is not None:
                pooled_projections = split_forward_gather_backward(
                    pooled_projections, dim=0, process_group=get_cfg_group()
                )

        if image_rotary_emb is not None:
            freqs_cos, freqs_sin = image_rotary_emb

            def get_rotary_emb_chunk(freqs):
                dim_thw = freqs.shape[-1]
                freqs = freqs.reshape(temporal_size, -1, dim_thw)
                freqs = torch.chunk(freqs, get_sp_group().world_size, dim=-2)[get_sp_group().rank_in_group]
                freqs = freqs.reshape(-1, dim_thw)
                return freqs

            freqs_cos = get_rotary_emb_chunk(freqs_cos)
            freqs_sin = get_rotary_emb_chunk(freqs_sin)
            image_rotary_emb = (freqs_cos, freqs_sin)

        improved_kwargs = {}
        signatures = inspect.signature(original_forward).parameters
        if 'pooled_projections' in signatures:
            improved_kwargs['pooled_projections'] = pooled_projections

        output = original_forward(
            hidden_states,
            encoder_hidden_states,
            timestep=timestep,
            image_rotary_emb=image_rotary_emb,
            *args,
            **improved_kwargs,
            **kwargs,
        )

        return_dict = not isinstance(output, tuple)
        sample = output[0]
        sample = gather_forward_split_backward(sample, dim=-2, process_group=get_sp_group())

        if get_cfg_group().world_size > 1:
            sample = gather_forward_split_backward(sample, dim=0, process_group=get_cfg_group())

        if return_dict:
            return output.__class__(sample, *output[1:])
        return (sample, *output[1:])

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward

    original_patch_embed_forward = transformer.patch_embed.forward

    @functools.wraps(transformer.patch_embed.__class__.forward)
    def new_patch_embed(
        self, text_embeds: torch.Tensor, image_embeds: torch.Tensor
    ):
        text_embeds = gather_forward_split_backward(
            text_embeds, dim=-2, process_group=get_sp_group()
        )
        image_embeds = gather_forward_split_backward(
            image_embeds, dim=-2, process_group=get_sp_group()
        )
        batch, num_frames, _, _, _ = image_embeds.shape
        text_len = text_embeds.shape[-2]

        output = original_patch_embed_forward(text_embeds, image_embeds)

        text_embeds = output[:,:text_len,:]
        if self.patch_size_t is None:
            image_embeds = output[:,text_len:,:].reshape(batch, num_frames, -1, output.shape[-1])
        else:
            image_embeds = output[:,text_len:,:].reshape(batch, num_frames // self.patch_size_t, -1, output.shape[-1])

        text_embeds = split_forward_gather_backward(
            text_embeds, dim=-2, process_group=get_sp_group()
        )
        image_embeds = split_forward_gather_backward(
            image_embeds, dim=-2, process_group=get_sp_group()
        )
        image_embeds = image_embeds.reshape(batch, -1, image_embeds.shape[-1])
        return torch.cat([text_embeds, image_embeds], dim=1)

    new_patch_embed = new_patch_embed.__get__(transformer.patch_embed)
    transformer.patch_embed.forward = new_patch_embed
