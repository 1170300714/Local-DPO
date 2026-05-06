import functools
from typing import Optional

import torch

from diffusers.models.autoencoders.autoencoder_kl_cogvideox import (
    DecoderOutput,
    CogVideoXCausalConv3d
)

from extensions.xfuser.core.distributed.parallel_state import get_mp_group


def parallelize_vae(vae):

    setattr(vae, "_enable_context_parallel", True)

    def enable_context_parallel(self):
        setattr(self, "_enable_context_parallel", True)
        for name, module in self.named_modules():
            if isinstance(module, CogVideoXCausalConv3d):
                setattr(module, "_enable_context_parallel", True)
    vae.__class__.enable_context_parallel = enable_context_parallel

    def disable_context_parallel(self):
        setattr(self, "_enable_context_parallel", False)
        for name, module in self.named_modules():
            if isinstance(module, CogVideoXCausalConv3d):
                setattr(module, "_enable_context_parallel", False)
    vae.__class__.disable_context_parallel = disable_context_parallel

    def clear_cache_padding(vae):
        for name, module in vae.named_modules():
            if isinstance(module, CogVideoXCausalConv3d) and hasattr(module, "cache_padding"):
                delattr(module, "cache_padding")

    def new_tiled_encode(self, x: torch.Tensor) -> torch.Tensor:

        enable_parallel = getattr(self, "_enable_context_parallel", True)

        batch_size, num_channels, num_frames, height, width = x.shape

        overlap_height = int(self.tile_sample_min_height * (1 - self.tile_overlap_factor_height))
        overlap_width = int(self.tile_sample_min_width * (1 - self.tile_overlap_factor_width))
        blend_extent_height = int(self.tile_latent_min_height * self.tile_overlap_factor_height)
        blend_extent_width = int(self.tile_latent_min_width * self.tile_overlap_factor_width)
        row_limit_height = self.tile_latent_min_height - blend_extent_height
        row_limit_width = self.tile_latent_min_width - blend_extent_width
        frame_batch_size = self.num_sample_frames_batch_size

        rows = []
        for i in range(0, height, overlap_height):
            row = []
            for j in range(0, width, overlap_width):
                num_batches = max(num_frames // frame_batch_size, 1)
                time = []

                if num_frames == 1:
                    x_temporal_tiles = [x]
                    enable_parallel = False
                else:
                    split_sections = [frame_batch_size + num_frames % frame_batch_size] + [frame_batch_size] * (num_batches - 1)
                    x_temporal_tiles = list(torch.split(x, split_sections, dim=2))
                world_size = get_mp_group().world_size if enable_parallel else 1
                num_chunks = (len(x_temporal_tiles) + world_size - 1) // world_size
                for k in range(num_chunks):
                    chunk = x_temporal_tiles[k * world_size: (k + 1) * world_size]
                    original_chunk_size = len(chunk)
                    if original_chunk_size < world_size:
                        chunk.extend(x_temporal_tiles[: world_size - original_chunk_size])

                    chunk_idx = get_mp_group().rank_in_group if enable_parallel else 0
                    local_tile = chunk[chunk_idx][
                        :,
                        :,
                        :,
                        i : i + self.tile_sample_min_height,
                        j : j + self.tile_sample_min_width,
                    ]
                    local_output, _ = self.encoder(local_tile)
                    if self.quant_conv is not None:
                        local_output = self.quant_conv(local_output)
                    latent_batch_size = frame_batch_size // self.config.temporal_compression_ratio

                    if enable_parallel:
                        all_local_outputs = get_mp_group().all_gather(
                            local_output[:, :, -latent_batch_size:].contiguous(), dim=2, separate_tensors=True)[:original_chunk_size]
                        if isinstance(all_local_outputs, torch.Tensor):
                            all_local_outputs = [all_local_outputs]
                        if chunk[0].shape[2] > frame_batch_size:
                            first_extra_latents = torch.empty_like(local_output[:, :, :1])
                            if get_mp_group().is_first_rank:
                                first_extra_latents = local_output[:, :, :-latent_batch_size].contiguous()
                            first_extra_latents = get_mp_group().broadcast(first_extra_latents, src=0)
                            all_local_outputs = [first_extra_latents] + all_local_outputs
                        output = torch.cat(all_local_outputs, dim=2)
                    else:
                        output = local_output

                    time.append(output)

                clear_cache_padding(self)

                row.append(torch.cat(time, dim=2))
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent_width)
                result_row.append(tile[:, :, :, :row_limit_height, :row_limit_width])
            result_rows.append(torch.cat(result_row, dim=4))
        enc = torch.cat(result_rows, dim=3)
        return enc

    @functools.wraps(vae.__class__._encode)
    def new_inner_encode(self, x: torch.Tensor):
        batch_size, num_channels, num_frames, height, width = x.shape

        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            return new_tiled_encode(self, x)

        enable_parallel = getattr(self, "_enable_context_parallel", True)

        frame_batch_size = self.num_sample_frames_batch_size
        num_batches = max(num_frames // frame_batch_size, 1)
        enc = []

        if num_frames == 1:
            x_temporal_tiles = [x]
            enable_parallel = False
        else:
            split_sections = [frame_batch_size + num_frames % frame_batch_size] + [frame_batch_size] * (num_batches - 1)
            x_temporal_tiles = list(torch.split(x, split_sections, dim=2))
        world_size = get_mp_group().world_size if enable_parallel else 1
        num_chunks = (len(x_temporal_tiles) + world_size - 1) // world_size
        for k in range(num_chunks):
            chunk = x_temporal_tiles[k * world_size: (k + 1) * world_size]
            original_chunk_size = len(chunk)
            if original_chunk_size < world_size:
                chunk.extend(x_temporal_tiles[: world_size - original_chunk_size])

            chunk_idx = get_mp_group().rank_in_group if enable_parallel else 0
            local_slice = chunk[chunk_idx]
            local_output, _ = self.encoder(local_slice)
            if self.quant_conv is not None:
                local_output = self.quant_conv(local_output)

            if enable_parallel:
                latent_batch_size = frame_batch_size // self.config.temporal_compression_ratio
                all_local_outputs = get_mp_group().all_gather(
                    local_output[:, :, -latent_batch_size:].contiguous(), dim=2, separate_tensors=True)[:original_chunk_size]
                if isinstance(all_local_outputs, torch.Tensor):
                    all_local_outputs = [all_local_outputs]
                if chunk[0].shape[2] > frame_batch_size:
                    first_extra_latents = torch.empty_like(local_output[:, :, :1])
                    if get_mp_group().is_first_rank:
                        first_extra_latents = local_output[:, :, :-latent_batch_size].contiguous()
                    first_extra_latents = get_mp_group().broadcast(first_extra_latents, src=0)
                    all_local_outputs = [first_extra_latents] + all_local_outputs
                output = torch.cat(all_local_outputs, dim=2)
            else:
                output = local_output
            enc.append(output)

        enc = torch.cat(enc, dim=2)
        clear_cache_padding(self)
        return enc

    new_inner_encode = new_inner_encode.__get__(vae)
    vae._encode = new_inner_encode

    def new_tiled_decode(self, z: torch.Tensor, return_dict: bool = True) -> torch.Tensor:

        enable_parallel = getattr(self, "_enable_context_parallel", True)

        batch_size, num_channels, num_latents, height, width = z.shape

        overlap_height = int(self.tile_latent_min_height * (1 - self.tile_overlap_factor_height))
        overlap_width = int(self.tile_latent_min_width * (1 - self.tile_overlap_factor_width))
        blend_extent_height = int(self.tile_sample_min_height * self.tile_overlap_factor_height)
        blend_extent_width = int(self.tile_sample_min_width * self.tile_overlap_factor_width)
        row_limit_height = self.tile_sample_min_height - blend_extent_height
        row_limit_width = self.tile_sample_min_width - blend_extent_width
        latent_batch_size = self.num_latent_frames_batch_size

        rows = []
        for i in range(0, height, overlap_height):
            row = []
            for j in range(0, width, overlap_width):
                num_batches = max(num_latents // latent_batch_size, 1)
                time = []

                if num_latents == 1:
                    z_temporal_tiles = [z]
                    enable_parallel = False
                else:
                    split_sections = [latent_batch_size + num_latents % latent_batch_size] + [latent_batch_size] * (num_batches - 1)
                    z_temporal_tiles = list(torch.split(z, split_sections, dim=2))
                world_size = get_mp_group().world_size if enable_parallel else 1
                num_chunks = (len(z_temporal_tiles) + world_size - 1) // world_size
                for k in range(num_chunks):
                    chunk = z_temporal_tiles[k * world_size: (k + 1) * world_size]
                    original_chunk_size = len(chunk)
                    if original_chunk_size < world_size:
                        chunk.extend(z_temporal_tiles[: world_size - original_chunk_size])

                    chunk_idx = get_mp_group().rank_in_group if enable_parallel else 0
                    local_tile = chunk[chunk_idx][
                        :,
                        :,
                        :,
                        i : i + self.tile_latent_min_height,
                        j : j + self.tile_latent_min_width,
                    ]
                    if self.post_quant_conv is not None:
                        local_tile = self.post_quant_conv(local_tile)
                    local_output, _ = self.decoder(local_tile)

                    if enable_parallel:
                        frame_batch_size = latent_batch_size * self.config.temporal_compression_ratio
                        all_local_outputs = get_mp_group().all_gather(
                            local_output[:, :, -frame_batch_size:].contiguous(), dim=2, separate_tensors=True)[:original_chunk_size]
                        if isinstance(all_local_outputs, torch.Tensor):
                            all_local_outputs = [all_local_outputs]
                        if chunk[0].shape[2] > latent_batch_size:
                            first_extra_frames = torch.empty_like(local_output[:, :, :1])
                            if get_mp_group().is_first_rank:
                                first_extra_frames = local_output[:, :, :-frame_batch_size].contiguous()
                            first_extra_frames = get_mp_group().broadcast(first_extra_frames, src=0)
                            all_local_outputs = [first_extra_frames] + all_local_outputs
                        output = torch.cat(all_local_outputs, dim=2)
                    else:
                        output = local_output

                    time.append(output)

                clear_cache_padding(self)

                row.append(torch.cat(time, dim=2))
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent_width)
                result_row.append(tile[:, :, :, :row_limit_height, :row_limit_width])
            result_rows.append(torch.cat(result_row, dim=4))

        dec = torch.cat(result_rows, dim=3)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @functools.wraps(vae.__class__._decode)
    def new_inner_decode(self, z: torch.Tensor, return_dict: bool = True):
        batch_size, num_channels, num_latents, height, width = z.shape

        if self.use_tiling and (width > self.tile_latent_min_width or height > self.tile_latent_min_height):
            return new_tiled_decode(self, z, return_dict=return_dict)

        enable_parallel = getattr(self, "_enable_context_parallel", True)

        latent_batch_size = self.num_latent_frames_batch_size
        num_batches = max(num_latents // latent_batch_size, 1)
        dec = []

        if num_latents == 1:
            z_temporal_tiles = [z]
            enable_parallel = False
        else:
            split_sections = [latent_batch_size + num_latents % latent_batch_size] + [latent_batch_size] * (num_batches - 1)
            z_temporal_tiles = list(torch.split(z, split_sections, dim=2))
        world_size = get_mp_group().world_size if enable_parallel else 1
        num_chunks = (len(z_temporal_tiles) + world_size - 1) // world_size
        for k in range(num_chunks):
            chunk = z_temporal_tiles[k * world_size: (k + 1) * world_size]
            original_chunk_size = len(chunk)
            if original_chunk_size < world_size:
                chunk.extend(z_temporal_tiles[: world_size - original_chunk_size])

            chunk_idx = get_mp_group().rank_in_group if enable_parallel else 0
            local_slice = chunk[chunk_idx]
            if self.post_quant_conv is not None:
                local_slice = self.post_quant_conv(local_slice)
            local_output, _ = self.decoder(local_slice)

            if enable_parallel:
                frame_batch_size = latent_batch_size * self.config.temporal_compression_ratio
                all_local_outputs = get_mp_group().all_gather(
                    local_output[:, :, -frame_batch_size:].contiguous(), dim=2, separate_tensors=True)[:original_chunk_size]
                if isinstance(all_local_outputs, torch.Tensor):
                    all_local_outputs = [all_local_outputs]
                if chunk[0].shape[2] > latent_batch_size:
                    first_extra_frames = torch.empty_like(local_output[:, :, :1])
                    if get_mp_group().is_first_rank:
                        first_extra_frames = local_output[:, :, :-frame_batch_size].contiguous()
                    first_extra_frames = get_mp_group().broadcast(first_extra_frames, src=0)
                    all_local_outputs = [first_extra_frames] + all_local_outputs
                output = torch.cat(all_local_outputs, dim=2)
            else:
                output = local_output
            dec.append(output)

        dec = torch.cat(dec, dim=2)
        clear_cache_padding(self)

        if not return_dict:
            return (output,)
        return DecoderOutput(sample=output)

    new_inner_decode = new_inner_decode.__get__(vae)
    vae._decode = new_inner_decode

    @functools.wraps(CogVideoXCausalConv3d.forward)
    def new_conv_forward(self, inputs: torch.Tensor, conv_cache: Optional[torch.Tensor] = None) -> torch.Tensor:

        def context_parallel_padding(
            inputs: torch.Tensor, cache_padding: Optional[torch.Tensor],
            pad_mode, time_causal_padding, kernel_size,
        ) -> torch.Tensor:
            if pad_mode == "replicate":
                return torch.nn.functional.pad(inputs, time_causal_padding, mode="replicate")
            if kernel_size == 1:
                return inputs

            inputs = inputs.transpose(0, 2).contiguous()

            enable_parallel = getattr(self, "_enable_context_parallel", True)
            if inputs.shape[0] == 1:
                enable_parallel = False

            if enable_parallel and not get_mp_group().is_last_rank:
                get_mp_group().isend(inputs[-kernel_size + 1 :].contiguous())

            if enable_parallel and not get_mp_group().is_first_rank:
                recv_buffer = get_mp_group().recv(inputs[-kernel_size + 1 :].shape, dtype=inputs.dtype)
            else:
                if cache_padding is not None:
                    cached_inputs = [cache_padding.transpose(0, 2).contiguous()]
                else:
                    cached_inputs = [inputs[:1]] * (kernel_size - 1)
                recv_buffer = torch.cat(cached_inputs)

            inputs = torch.cat([recv_buffer, inputs])

            return inputs.transpose(0, 2).contiguous()

        self.cache_padding = getattr(self, 'cache_padding', None)
        inputs = context_parallel_padding(inputs, self.cache_padding,
            self.pad_mode, self.time_causal_padding, self.time_kernel_size)

        if self.pad_mode == "replicate":
            new_conv_cache = None
        else:
            padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad)
            new_conv_cache = inputs[:, :, -self.time_kernel_size + 1 :].clone()
            inputs = torch.nn.functional.pad(inputs, padding_2d, mode="constant", value=0)

        del self.cache_padding
        self.cache_padding = None
        world_size = get_mp_group().world_size if getattr(self, "_enable_context_parallel", True) else 1
        if world_size == 1:
            self.cache_padding = new_conv_cache
        elif new_conv_cache is not None and self.time_kernel_size > 1:
            if get_mp_group().is_last_rank:
                get_mp_group().isend(new_conv_cache.contiguous())
            if get_mp_group().is_first_rank:
                self.cache_padding = get_mp_group().recv(new_conv_cache.shape, dtype=new_conv_cache.dtype)

        output = self.conv(inputs)
        return output, new_conv_cache

    for name, module in vae.named_modules():
        if isinstance(module, CogVideoXCausalConv3d):
            setattr(module, "_enable_context_parallel", True)
            new_forward = new_conv_forward.__get__(module)
            module.forward = new_forward
