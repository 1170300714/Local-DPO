import torch
import numpy as np


def vae_encode(x, vae, dtype):

    x = x.permute(0, 2, 1, 3, 4) 
    dist = vae.encode(x).latent_dist
    latent = dist.sample() * vae.config.scaling_factor
    latent = latent.permute(0, 2, 1, 3, 4) 
    return latent.to(memory_format=torch.contiguous_format, dtype=dtype)


def prepare_visual_condition(
    content: torch.Tensor,
    latent: torch.Tensor,
    condition_config: dict,
    vae: torch.nn.Module,
    dtype: torch.dtype,
):

    B, F, C, H, W = content.shape


    _, lT, _, lH, lW = latent.shape

    masks = torch.zeros(B, lT, 1, lH, lW, dtype=content.dtype, device=content.device)
    cond_latent = torch.zeros_like(latent)

    if F == 1: 
        return torch.cat((masks, cond_latent), dim=2), ['t2i', ] * B

    cond_type_probs = np.array(list(condition_config.values())) / np.sum(list(condition_config.values()))

    cond_type_list = []
    for i in range(B):
        cond_type = np.random.choice(
            list(condition_config.keys()),
            p=cond_type_probs,
        )
        cond_type_list.append(cond_type)
        if cond_type == 't2v':
            continue
        if cond_type == 'i2v_head':
            masks[i, 0, :, :, :] = 1
            cond_latent[i] = vae_encode(
                torch.concat([
                    content[i:i+1, :1, :, :, :].cpu(),
                    torch.zeros(1, F - 1, C, H, W)
                ], dim=1).to(latent.device, dtype=dtype),
                vae, dtype,
            )[0]
        elif cond_type == 'i2v_tail':
            masks[i, -1, :, :, :] = 1
            cond_latent[i] = vae_encode(
                torch.concat([
                    torch.zeros(1, F - 1, C, H, W),
                    content[i:i+1, -1:, :, :, :].cpu(),
                ], dim=1).to(latent.device, dtype=dtype),
                vae, dtype,
            )[0]
        elif cond_type == 'i2v_loop':
            masks[i, 0, :, :, :] = 1
            masks[i, -1, :, :, :] = 1
            cond_latent[i] = vae_encode(
                torch.concat([
                    content[i:i+1, :1, :, :, :].cpu(),
                    torch.zeros(1, F - 2, C, H, W),
                    content[i:i+1, -1:, :, :, :].cpu(),
                ], dim=1).to(latent.device, dtype=dtype),
                vae, dtype,
            )[0]
    return torch.cat((masks, cond_latent), dim=2), cond_type_list


def prepare_visual_condition_dpo(
    content: torch.Tensor,
    latent: torch.Tensor,
    condition_config: dict,
    vae: torch.nn.Module,
    dtype: torch.dtype,
):

    B, F, C, H, W = content.shape

    real_video_batch = int(B // 2)

  
    _, lT, _, lH, lW = latent.shape

    masks = torch.zeros(B, lT, 1, lH, lW, dtype=content.dtype, device=content.device)
    cond_latent = torch.zeros_like(latent)

    if F == 1: 
        return torch.cat((masks, cond_latent), dim=2), ['t2i', ] * B  

    cond_type_probs = np.array(list(condition_config.values())) / np.sum(list(condition_config.values()))

    cond_type_pool = list(condition_config.keys())
    supported_cond_types = ['t2v', 'i2v_head']
    extra_cond_types = list(set(cond_type_pool).difference(set(supported_cond_types)))
    assert len(extra_cond_types) == 0, f'Condition type: {extra_cond_types} are not supported here'

    cond_type_list = []
    for i in range(real_video_batch):  
        cond_type = np.random.choice(
            list(condition_config.keys()),
            p=cond_type_probs,
        )
        cond_type_list.append(cond_type)
        if cond_type == 't2v':
            continue
        elif cond_type == 'i2v_head':
            masks[i, 0, :, :, :] = 1
            masks[i+real_video_batch, 0, :, :, :] = 1
            cond_latent[i] = vae_encode(
                torch.concat([
                    content[i:i+1, :1, :, :, :].cpu(),
                    torch.zeros(1, F - 1, C, H, W)
                ], dim=1).to(latent.device, dtype=dtype),
                vae, dtype,
            )[0]
            cond_latent[i] = vae_encode(
                torch.concat([
                    content[i+real_video_batch:i+real_video_batch+1, :1, :, :, :].cpu(),
                    torch.zeros(1, F - 1, C, H, W)
                ], dim=1).to(latent.device, dtype=dtype),
                vae, dtype,
            )[0]
        else:
            raise ValueError(f'Condition type: {cond_type} is not supported here.')
    return torch.cat((masks, cond_latent), dim=2), cond_type_list