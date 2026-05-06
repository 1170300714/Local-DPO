import math
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.distributed as dist
from collections import defaultdict
from diffusers.models.attention import Attention


r'''Usage
```python
    with AttentionMapCollector(transformer) as collector:
        transformer(...)
    attnmap = collector.get_attention_map() # [B, N, N], N is the total number of visual and textual tokens
```
'''


@torch.no_grad()
def compute_attention_map(q, k, module_name=None, text_length=226):

    w = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    w = torch.softmax(w, dim=-1)
    w = w.mean(dim=1).detach().cpu().float() 



    return w


class AttentionMapCollector:

    def __init__(self, model: torch.nn.Module):
        self.buffers = {}
        self.model = model.module if hasattr(model, "module") else model

        self.attach_hook()

    def attach_hook(self):
        self.buffers.clear()

        def attention_map_collect_wrapper(func, key):
            def inner_call(q, k, v, *args, **kwargs):
                self.buffers[key] = compute_attention_map(q, k)
                return func(q, k, v, *args, **kwargs)
            return inner_call

        for key, module in self.model.named_modules():
            if isinstance(module, Attention):
                if hasattr(module.processor, "__original_attn_func"): continue
                assert hasattr(module.processor, "attn_func"), \
                    "AttentionMapCollector only supports AttnProcessor with attn_func interface"
                setattr(module.processor, "__original_attn_func", module.processor.attn_func)
                module.processor.attn_func = attention_map_collect_wrapper(
                    module.processor.attn_func,
                    key,
                )

    def detach_hook(self):
        for _, module in self.model.named_modules():
            if isinstance(module, Attention):
                if not hasattr(module.processor, "__original_attn_func"): continue
                assert hasattr(module.processor, "attn_func"), \
                    "AttentionMapCollector only supports AttnProcessor with attn_func interface"
                module.processor.attn_func = getattr(module.processor, "__original_attn_func")
                delattr(module.processor, "__original_attn_func")

    def get_attention_map(self, key=None):
        if key is None:
            return torch.mean(torch.stack(list(self.buffers.values())), dim=0)
        return self.buffers[key]

    def __enter__(self):
        self.attach_hook()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.detach_hook()


    @classmethod
    def visualize_vision_to_text_attention_masks(cls, attn_maps, token_indices, frame_idx, tokens=None, figsize_scale=1):
        """
        Visualize text-token-wise visual attention map

        Args:
            attn_maps: torch.Tensor - Attention maps in shape of [B, F, H, W, T]
            token_indices: List[List[int]] - List of token indices, e.g. [[0], [1], [2], [0,1,2]]
            frame_idx: int - index of latent frame to visualize
            tokens: Optional[List[str]] - List of tokens. If provided, tokens will be used as subplot titles
            figsize_scale: float - Scale of figure size
        """
        def apply_magma_colormap(image):
            """
            Convert grayscale image to magma colormap
            """
            img_array = np.array(image)
            img_normalized = img_array / 255.0
            colored_array = plt.cm.magma(img_normalized)
            colored_image = Image.fromarray((colored_array[:, :, :3] * 255).astype(np.uint8))
            return colored_image

        num_tokens = len(token_indices)

  
        n_cols = min(num_tokens, 4)
        n_rows = (num_tokens + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * figsize_scale, n_rows * figsize_scale))
        axes = [axes] if num_tokens == 1 else axes.flatten()
        subplot_adjust = {'wspace': 0.05, 'hspace': 0}

 
        fig.patch.set_facecolor('white')

    
        for idx, token_idx in enumerate(token_indices):
     
            token_idx = token_idx if isinstance(token_idx, list) else [token_idx]
            m = attn_maps[..., token_idx].sum(dim=-1)  
 
            m_min = m.amin(dim=(2, 3), keepdim=True)
            m_max = m.amax(dim=(2, 3), keepdim=True)
            mask = (m - m_min) / (m_max - m_min + 1e-6)
    
   
            frame_mask = mask[0, frame_idx]

      
            colored_frame = apply_magma_colormap(Image.fromarray((frame_mask.cpu().numpy() * 255).astype(np.uint8)))

   
            axes[idx].imshow(np.array(colored_frame))
            axes[idx].axis('off')
            if tokens is not None:
                token = tokens[idx] if idx < len(tokens) else '<pad>'
                axes[idx].set_title(token)


        plt.subplots_adjust(**subplot_adjust)

        return fig



def recursive_defaultdict_to_dict(d):
    if not isinstance(d, dict): return d
    nd = {}
    for k, v in d.items():
        nd[k] = recursive_defaultdict_to_dict(v)
    return nd


class FrequencyCollector:

    def __init__(self):
        self.buffers = defaultdict(lambda: defaultdict(int))

    def load_state_dict(self, state_dict):
        for tag, tag_buffers in state_dict.items():
            for value, count in tag_buffers.items():
                self.buffers[tag][value] = count

    def add_single(self, value, count, tag=None):
        tag = tag or "default"
        if not isinstance(value, str):
            value = str(value)
        self.buffers[tag][value] += count

    def add_multiple(self, values, counts, tag=None):
        for value, count in zip(values, counts):
            self.add_single(value, count, tag)

    def add_values(self, values, tag=None):
        for value in values:
            self.add_single(value, 1, tag)

    def add_values_with_bins(self, values, bins, tag=None):
        for value in values:
            f = list(filter(lambda x: x < value, bins))
            left_border = max(f) if len(f) else bins[0]
            self.add_single(left_border, 1, tag)

    def reset(self):
        self.buffers.clear()

    def get_collections(self, distributed_gather=False):
        if not distributed_gather or not dist.is_initialized() or dist.get_world_size() == 1:
            return recursive_defaultdict_to_dict(self.buffers)

        buffers = recursive_defaultdict_to_dict(self.buffers)
        buffers_list = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(buffers_list, buffers)

        merged_buffers = defaultdict(lambda: defaultdict(int))
        for buffers in buffers_list:
            for tag, tag_buffers in buffers.items():
                for value, count in tag_buffers.items():
                    merged_buffers[tag][value] += count

        return recursive_defaultdict_to_dict(merged_buffers)
