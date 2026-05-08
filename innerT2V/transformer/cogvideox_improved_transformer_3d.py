import torch

from diffusers.models.embeddings import apply_rotary_emb

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