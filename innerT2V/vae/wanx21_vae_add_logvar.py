from typing import List, Optional, Tuple, Union

import math
import torch
from diffusers import AutoencoderKLWan
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.configuration_utils import register_to_config


from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution


class AutoencoderKLWanImproved(AutoencoderKLWan):

    @register_to_config
    def __init__(
        self,
        base_dim: int = 96,
        z_dim: int = 16,
        dim_mult: Tuple[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: List[float] = [],
        temperal_downsample: List[bool] = [False, True, True],
        dropout: float = 0.0,
        latents_mean: List[float] = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ],
        latents_std: List[float] = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ],
    ) -> None:
        super().__init__(base_dim, z_dim, dim_mult, num_res_blocks, attn_scales, temperal_downsample, dropout, latents_mean, latents_std)
        self.mean = torch.tensor(latents_mean)
        self.std = torch.tensor(latents_std)
        self.scale = [self.mean, 1.0 / self.std]

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        self.clear_cache()
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = torch.cat([out, out_], 2)

        enc = self.quant_conv(out)
        mu, logvar = enc[:, : self.z_dim, :, :, :], enc[:, self.z_dim :, :, :, :]
        if isinstance(self.scale[0], torch.Tensor):
            self.scale[0] = self.scale[0].to(dtype=mu.dtype, device=mu.device)
            self.scale[1] = self.scale[1].to(dtype=mu.dtype, device=mu.device)
            mu = (mu - self.scale[0].view(1, self.z_dim, 1, 1, 1)) * self.scale[1].view(
                1, self.z_dim, 1, 1, 1)
            logvar = logvar + 2 * torch.log(self.scale[1].view(1, self.z_dim, 1, 1, 1))
        else:
            mu = (mu - self.scale[0]) * self.scale[1]
            logvar = logvar + 2 * math.log(self.scale[1])
        enc = torch.cat([mu, logvar], dim=1)
        self.clear_cache()
        return enc
    

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:

        h = self._encode(x)
        posterior = DiagonalGaussianDistribution(h)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)