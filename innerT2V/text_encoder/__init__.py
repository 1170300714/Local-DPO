import torch
from typing import Optional
from contextlib import nullcontext

from .t5 import encode_t5_prompt



def compute_prompt_embeddings(
    tokenizer,
    text_encoder,
    prompt: str,
    max_sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
    prompt_2: Optional[str] = None,
    requires_grad: bool = False,
    offload_model: bool = True,
):
    text_encoder.to(device)
    with torch.no_grad() if not requires_grad else nullcontext():
        prompt_embeds = encode_t5_prompt(
            tokenizer,
            text_encoder,
            prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
    if offload_model:
        text_encoder.to('cpu')
  

    pooled_prompt_embeds = None


    return prompt_embeds, pooled_prompt_embeds
