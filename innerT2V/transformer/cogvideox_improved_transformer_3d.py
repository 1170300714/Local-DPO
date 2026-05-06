from typing import Any, Dict, Optional, Tuple, Union

import torch

from diffusers.configuration_utils import register_to_config
from diffusers.models.transformers.cogvideox_transformer_3d import (
    CogVideoXTransformer3DModel,
    CogVideoXBlock,
)
from diffusers.models.embeddings import (
    TimestepEmbedding, Timesteps,
    PixArtAlphaTextProjection,
    apply_rotary_emb,
)
from diffusers.models.attention import Attention
from diffusers.models.normalization import LayerNorm

from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from utils.torch_utils import zero_module
from utils.logger import get_logger


logger = get_logger()

class CogVideoXUnifyAttnProcessor2_0:

    def __init__(self, attn_func=None):
        if attn_func is None:
            attn_func = torch.nn.functional.scaled_dot_product_attention
        self.attn_func = attn_func

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states,
        attention_mask = None,
        image_rotary_emb = None,
    ) -> torch.Tensor:
        batch_size, text_seq_length = encoder_hidden_states.shape[:2]

        if getattr(attn, 'add_q_proj') is None:
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            if getattr(attn, 'add_q_proj') is None:
                query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
                if not attn.is_cross_attention:
                    key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)
            else:
                query = apply_rotary_emb(query, image_rotary_emb)
                if not attn.is_cross_attention:
                    key = apply_rotary_emb(key, image_rotary_emb)

        if getattr(attn, 'add_q_proj') is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_key = encoder_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_value = encoder_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if getattr(attn, 'norm_added_q') is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if getattr(attn, 'norm_added_k') is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=2)
            key = torch.cat([encoder_key, key], dim=2)
            value = torch.cat([encoder_value, value], dim=2)

        hidden_states = self.attn_func(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        if getattr(attn, 'to_add_out') is None:
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states, hidden_states = hidden_states.split(
                [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
            )
        else:
            encoder_hidden_states, hidden_states = hidden_states.split(
                [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
            )

            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


class CogVideoXCrossAttnProcessor2_0(CogVideoXUnifyAttnProcessor2_0):

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states,
        attention_mask = None,
        image_rotary_emb = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        hidden_states = self.attn_func(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class Gate(torch.nn.Module):

    def __init__(self, dim: Tuple[int]):
        super().__init__()

        if isinstance(dim, int):
            dim = (dim, )
        self.dim = dim
        self.weight = torch.nn.Parameter(torch.zeros(*dim))

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.copy_(torch.randn(*self.dim) / self.dim[-1]**0.5)

    def forward(self, x):
        return x * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class CogVideoXSeperateBlock(CogVideoXBlock):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__(
            dim,
            num_attention_heads,
            attention_head_dim,
            time_embed_dim,
            dropout,
            activation_fn,
            attention_bias,
            qk_norm,
            norm_elementwise_affine,
            norm_eps,
            final_dropout,
            ff_inner_dim,
            ff_bias,
            attention_out_bias,
        )

        self.norm_cross = LayerNorm(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.gate_cross = Gate((1, 1, dim))
        self.attn_cross = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXCrossAttnProcessor2_0(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)
        attention_kwargs = attention_kwargs or {}

        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **attention_kwargs,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        attn_hidden_states = self.attn_cross(
            hidden_states=self.norm_cross(hidden_states),
            encoder_hidden_states=encoder_hidden_states,
            **attention_kwargs,
        )
        hidden_states = hidden_states + self.gate_cross(attn_hidden_states)

        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states


class CondPatchEmbed(torch.nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        patch_size_t: Optional[int] = None,
        in_channels: int = 16,
        embed_dim: int = 1920,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.embed_dim = embed_dim

        if patch_size_t is None:
            self.proj = torch.nn.Conv2d(
                in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
            )
        else:
            self.proj = torch.nn.Linear(in_channels * patch_size * patch_size * patch_size_t, embed_dim)


    def forward(self, image_embeds: torch.Tensor):
        batch_size, num_frames, channels, height, width = image_embeds.shape

        if self.patch_size_t is None:
            image_embeds = image_embeds.reshape(-1, channels, height, width)
            image_embeds = self.proj(image_embeds)
            image_embeds = image_embeds.view(batch_size, num_frames, *image_embeds.shape[1:])
            image_embeds = image_embeds.flatten(3).transpose(2, 3)
            image_embeds = image_embeds.flatten(1, 2)
        else:
            p = self.patch_size
            p_t = self.patch_size_t

            image_embeds = image_embeds.permute(0, 1, 3, 4, 2)
            image_embeds = image_embeds.reshape(
                batch_size, num_frames // p_t, p_t, height // p, p, width // p, p, channels
            )
            image_embeds = image_embeds.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(4, 7).flatten(1, 3)
            image_embeds = self.proj(image_embeds)

        embeds = image_embeds.contiguous()

        return embeds


class CogVideoXImprovedTransformer3DModel(CogVideoXTransformer3DModel):

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        ofs_embed_dim: Optional[int] = None,
        clarify_embed_dim: Optional[int] = None,
        aes_embed_dim: Optional[int] = None,
        vmaf_embed_dim: Optional[int] = None,
        fps_embed_dim: Optional[int] = None,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        patch_size_t: Optional[int] = None,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        patch_bias: bool = True,
        use_text_qkvproj: bool = False,
        use_text_outproj: bool = False,
        pooled_projection_dim: Optional[int] = None,
        seperate_attention: bool = False,
        conditional_in_channels: Optional[int] = None,
    ):
        super().__init__(
            num_attention_heads,
            attention_head_dim,
            in_channels,
            out_channels,
            flip_sin_to_cos,
            freq_shift,
            time_embed_dim,
            ofs_embed_dim,
            text_embed_dim,
            num_layers,
            dropout,
            attention_bias,
            sample_width,
            sample_height,
            sample_frames,
            patch_size,
            patch_size_t,
            temporal_compression_ratio,
            max_text_seq_length,
            activation_fn,
            timestep_activation_fn,
            norm_elementwise_affine,
            norm_eps,
            spatial_interpolation_scale,
            temporal_interpolation_scale,
            use_rotary_positional_embeddings,
            use_learned_positional_embeddings,
            patch_bias,
        )

        inner_dim = num_attention_heads * attention_head_dim

        pooled_text_embedder = None
        if pooled_projection_dim is not None:
            pooled_text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, time_embed_dim, act_fn='silu')
        self.pooled_text_embedder = pooled_text_embedder

        clarify_proj = clarify_embedding = None
        if clarify_embed_dim is not None:
            clarify_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
            clarify_embedding = TimestepEmbedding(inner_dim, clarify_embed_dim, timestep_activation_fn)
        self.clarify_proj = clarify_proj
        self.clarify_embedding = clarify_embedding

        aes_proj = aes_embedding = None
        if aes_embed_dim is not None:
            aes_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
            aes_embedding = TimestepEmbedding(inner_dim, aes_embed_dim, timestep_activation_fn)
        self.aes_proj = aes_proj
        self.aes_embedding = aes_embedding

        vmaf_proj = vmaf_embedding = None
        if vmaf_embed_dim is not None:
            vmaf_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
            vmaf_embedding = TimestepEmbedding(inner_dim, vmaf_embed_dim, timestep_activation_fn)
        self.vmaf_proj = vmaf_proj
        self.vmaf_embedding = vmaf_embedding

        fps_proj = fps_embedding = None
        if fps_embed_dim is not None:
            fps_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
            fps_embedding = TimestepEmbedding(inner_dim, fps_embed_dim, timestep_activation_fn)
        self.fps_proj = fps_proj
        self.fps_embedding = fps_embedding

        if seperate_attention:
            for i in range(len(self.transformer_blocks)):
                block = CogVideoXSeperateBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                block.load_state_dict(self.transformer_blocks[i].state_dict(), strict=False)
                self.transformer_blocks[i] = block
        elif use_text_qkvproj or use_text_outproj:
            for block in self.transformer_blocks:
                attn = Attention(
                    query_dim=inner_dim,
                    dim_head=attention_head_dim,
                    heads=num_attention_heads,
                    qk_norm="layer_norm",
                    eps=1e-6,
                    bias=attention_bias,
                    out_bias=True,
                    added_kv_proj_dim=inner_dim if use_text_qkvproj else None,
                    added_proj_bias=attention_bias,
                    context_pre_only=not use_text_outproj,
                )
                attn.load_state_dict(block.attn1.state_dict(), strict=False)
                block.attn1 = attn

        self.cond_patchify = None
        if conditional_in_channels is not None and conditional_in_channels > 0:
            self.cond_patchify = CondPatchEmbed(
                patch_size=patch_size,
                patch_size_t=patch_size_t,
                in_channels=conditional_in_channels,
                embed_dim=inner_dim,
                bias=patch_bias,
            )

        self.init_additional_params()

        for block in self.transformer_blocks:
            block.attn1.processor = CogVideoXUnifyAttnProcessor2_0()

    def init_additional_params(self, tailed_zero_init: bool = True):
        if getattr(self, 'pooled_text_embedder') is not None:
            self.pooled_text_embedder.linear_1.reset_parameters()
            self.pooled_text_embedder.linear_2.reset_parameters()
            if tailed_zero_init:
                self.pooled_text_embedder.linear_2 = zero_module(self.pooled_text_embedder.linear_2)

        if getattr(self, 'clarify_embedding') is not None:
            self.clarify_embedding.linear_1.reset_parameters()
            self.clarify_embedding.linear_2.reset_parameters()
            if tailed_zero_init:
                self.clarify_embedding.linear_2 = zero_module(self.clarify_embedding.linear_2)

        if getattr(self, 'aes_embedding') is not None:
            self.aes_embedding.linear_1.reset_parameters()
            self.aes_embedding.linear_2.reset_parameters()
            if tailed_zero_init:
                self.aes_embedding.linear_2 = zero_module(self.aes_embedding.linear_2)

        if getattr(self, 'vmaf_embedding') is not None:
            self.vmaf_embedding.linear_1.reset_parameters()
            self.vmaf_embedding.linear_2.reset_parameters()
            if tailed_zero_init:
                self.vmaf_embedding.linear_2 = zero_module(self.vmaf_embedding.linear_2)

        if getattr(self, 'fps_embedding') is not None:
            self.fps_embedding.linear_1.reset_parameters()
            self.fps_embedding.linear_2.reset_parameters()
            if tailed_zero_init:
                self.fps_embedding.linear_2 = zero_module(self.fps_embedding.linear_2)

        for block in self.transformer_blocks:
            attn = block.attn1
            if getattr(attn, 'add_q_proj') is not None:
                attn.add_q_proj.load_state_dict(attn.to_q.state_dict())
            if getattr(attn, 'add_k_proj') is not None:
                attn.add_k_proj.load_state_dict(attn.to_k.state_dict())
            if getattr(attn, 'add_v_proj') is not None:
                attn.add_v_proj.load_state_dict(attn.to_v.state_dict())
            if getattr(attn, 'to_add_out') is not None:
                attn.to_add_out.load_state_dict(attn.to_out[0].state_dict())
            if getattr(attn, 'norm_added_q') is not None:
                attn.norm_added_q.load_state_dict(attn.norm_q.state_dict())
            if getattr(attn, 'norm_added_k') is not None:
                attn.norm_added_k.load_state_dict(attn.norm_k.state_dict())

            def recursive_reset_parameters(m):
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
                for sm in m.children():
                    recursive_reset_parameters(sm)

            if hasattr(block, 'attn_cross'):
                block.attn_cross.load_state_dict(block.attn1.state_dict(), strict=False)
                recursive_reset_parameters(block.norm_cross)
                recursive_reset_parameters(block.gate_cross)
                if tailed_zero_init:
                    block.gate_cross = zero_module(block.gate_cross)

        if getattr(self, 'cond_patchify') is not None:
            self.cond_patchify.proj.reset_parameters()
            if tailed_zero_init:
                self.cond_patchify.proj = zero_module(self.cond_patchify.proj)

    def reset_params(self, tailed_zero_init: bool = True):
        def recursive_reset_parameters(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
            for sm in m.children():
                recursive_reset_parameters(sm)
        recursive_reset_parameters(self)
        self.init_additional_params(tailed_zero_init)

    def enable_additional_params_grad(self):
        if getattr(self, 'pooled_text_embedder') is not None:
            self.pooled_text_embedder.requires_grad_(True)

        if getattr(self, 'clarify_embedding') is not None:
            self.clarify_embedding.requires_grad_(True)

        if getattr(self, 'aes_embedding') is not None:
            self.aes_embedding.requires_grad_(True)

        if getattr(self, 'vmaf_embedding') is not None:
            self.vmaf_embedding.requires_grad_(True)

        if getattr(self, 'fps_embedding') is not None:
            self.fps_embedding.requires_grad_(True)

        for block in self.transformer_blocks:
            attn = block.attn1
            if getattr(attn, 'add_q_proj') is not None:
                attn.add_q_proj.requires_grad_(True)
            if getattr(attn, 'add_k_proj') is not None:
                attn.add_k_proj.requires_grad_(True)
            if getattr(attn, 'add_v_proj') is not None:
                attn.add_v_proj.requires_grad_(True)
            if getattr(attn, 'to_add_out') is not None:
                attn.to_add_out.requires_grad_(True)
            if getattr(attn, 'norm_added_q') is not None:
                attn.norm_added_q.requires_grad_(True)
            if getattr(attn, 'norm_added_k') is not None:
                attn.norm_added_k.requires_grad_(True)

            def recursive_enable_grad(m):
                m.requires_grad_(True)
                for sm in m.children():
                    recursive_enable_grad(sm)

            if hasattr(block, 'attn_cross'):
                block.attn_cross.requires_grad_(True)
                recursive_enable_grad(block.norm_cross)
                recursive_enable_grad(block.gate_cross)

        if getattr(self, 'cond_patchify') is not None:
            self.cond_patchify.requires_grad_(True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        conditional_hidden_states: Optional[torch.Tensor] = None,
        pooled_projections: Optional[torch.Tensor] = None,
        clarify_score: Optional[Union[int, float, torch.LongTensor, torch.Tensor]] = None,
        aes_score: Optional[Union[int, float, torch.LongTensor, torch.Tensor]] = None,
        vmaf_score: Optional[Union[int, float, torch.LongTensor, torch.Tensor]] = None,
        fps_score: Optional[Union[int, float, torch.LongTensor, torch.Tensor]] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.clarify_embedding is not None and clarify_score is not None:
            clarity_emb = self.clarify_proj(clarify_score)
            clarity_emb = clarity_emb.to(dtype=hidden_states.dtype)
            clarity_emb = self.clarify_embedding(clarity_emb)
            emb = emb + clarity_emb

        if self.aes_embedding is not None and aes_score is not None:
            aes_emb = self.aes_proj(aes_score)
            aes_emb = aes_emb.to(dtype=hidden_states.dtype)
            aes_emb = self.aes_embedding(aes_emb)
            emb = emb + aes_emb

        if self.vmaf_embedding is not None and vmaf_score is not None:
            vmaf_emb = self.vmaf_proj(vmaf_score)
            vmaf_emb = vmaf_emb.to(dtype=hidden_states.dtype)
            vmaf_emb = self.vmaf_embedding(vmaf_emb)
            emb = emb + vmaf_emb

        if self.fps_embedding is not None and fps_score is not None:
            fps_emb = self.fps_proj(fps_score)
            fps_emb = fps_emb.to(dtype=hidden_states.dtype)
            fps_emb = self.fps_embedding(fps_emb)
            emb = emb + fps_emb

        if self.ofs_embedding is not None and ofs is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        if self.pooled_text_embedder is not None and pooled_projections is not None:
            txt_emb = self.pooled_text_embedder(pooled_projections)
            emb = emb + txt_emb

        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        if self.cond_patchify is not None and conditional_hidden_states is not None:
            conditional_hidden_states = self.cond_patchify(conditional_hidden_states)
            hidden_states = hidden_states + conditional_hidden_states

        if self.cond_patchify is not None and conditional_hidden_states is None:
            conditional_hidden_states = self.cond_patchify(
                hidden_states.new_zeros((batch_size, num_frames, channels + 1, height, width)))
            hidden_states = hidden_states + conditional_hidden_states

        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )

        if not self.config.use_rotary_positional_embeddings:
            hidden_states = self.norm_final(hidden_states)
        else:
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
