
import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import T5EncoderModel, T5Tokenizer

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.utils import is_torch_xla_available, logging
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps, CogVideoXPipeline

from text_encoder import compute_prompt_embeddings
from scheduler.scheduling_dpm_cogvideox_improved import CogVideoXDPMImprovedScheduler
from utils import prepare_rotary_positional_embeddings


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)


class CogVideoXImprovedPipeline(CogVideoXPipeline):

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModel,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],

    ):
        super().__init__(tokenizer, text_encoder, vae, transformer, scheduler)

        if not isinstance(scheduler, CogVideoXDPMImprovedScheduler):
            scheduler = CogVideoXDPMImprovedScheduler(**scheduler.config)
            self.register_modules(scheduler=scheduler)



    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, pooled_prompt_embeds = compute_prompt_embeddings(
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
                prompt=prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
                requires_grad=False,
            )
            _, seq_len, dimesion = prompt_embeds.shape
            prompt_embeds = prompt_embeds.unsqueeze(1).repeat(1, num_videos_per_prompt, 1, 1)
            prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, dimesion)

            if pooled_prompt_embeds is not None:
                pooled_prompt_embeds = pooled_prompt_embeds.unsqueeze(1).repeat(1, num_videos_per_prompt, 1)
                pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_videos_per_prompt, -1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds, negative_pooled_prompt_embeds = compute_prompt_embeddings(
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
                prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
                requires_grad=False,
            )

            _, seq_len, dimesion = negative_prompt_embeds.shape
            negative_prompt_embeds = negative_prompt_embeds.unsqueeze(1).repeat(1, num_videos_per_prompt, 1, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, dimesion)

            if negative_pooled_prompt_embeds is not None:
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.unsqueeze(1).repeat(
                    1, num_videos_per_prompt, 1
                )
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.view(batch_size * num_videos_per_prompt, -1)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def prepare_latents(
        self,
        batch_size,
        num_videos_per_prompt: int,
        num_channels_latents: int,
        num_frames: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        latents=None,
        condition=None,
    ):
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        patch_size_t = self.transformer.config.patch_size_t
        num_additional_latents = 0
        num_padded_frames = num_frames
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            num_additional_latents = patch_size_t - latent_frames % patch_size_t
            num_padded_frames += num_additional_latents * self.vae_scale_factor_temporal

        latents = super().prepare_latents(
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            num_frames=num_padded_frames,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=latents
        )

        if condition is None:
            return latents, None, num_additional_latents

        if isinstance(condition, dict):
            condition = [condition]

        if len(condition) != batch_size:
            raise ValueError(
                f"You have passed {len(condition)} conditions, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the conditions."
            )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        shape = (
            batch_size,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        T, H, W = num_frames, height, width
        B, lT, lC, lH, lW = shape

        masks = torch.zeros(B, lT, 1, lH, lW, dtype=dtype, device=device)
        cond_latent = torch.zeros(B, lT, lC, lH, lW, dtype=dtype, device=device)

        def prepare_image_tensor(image, h, w):
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=(h, w),
                mode="bilinear",
            )
            return image.unsqueeze(0)

        def vae_encode(x, vae, dtype, generator):
            x = x.permute(0, 2, 1, 3, 4)
            dist = vae.encode(x).latent_dist
            latent = dist.sample(generator) * vae.config.scaling_factor
            latent = latent.permute(0, 2, 1, 3, 4)
            return latent.to(memory_format=torch.contiguous_format, dtype=dtype)

        if T == 1:
            cond_latents = torch.cat((masks, cond_latent), dim=2)
        else:
            for i in range(B):
                cond = condition[i]
                cond_type = cond["cond_type"]
                gen = generator[i] if isinstance(generator, list) else generator
                if cond_type == 't2v':
                    continue
                if cond_type == 'i2v_head':
                    masks[i, 0, :, :, :] = 1
                    image = prepare_image_tensor(cond["first_image"], H, W)
                    cond_latent[i] = vae_encode(
                        torch.concat([
                            image.cpu(),
                            torch.zeros(1, T - 1, 3, H, W),
                        ], dim=1).to(device, dtype=dtype),
                        self.vae, dtype, gen,
                    )[0]
                elif cond_type == 'i2v_tail':
                    masks[i, -1, :, :, :] = 1
                    image = prepare_image_tensor(cond["last_image"], H, W)
                    cond_latent[i] = vae_encode(
                        torch.concat([
                            torch.zeros(1, T - 1, 3, H, W),
                            image.cpu(),
                        ], dim=1).to(device, dtype=dtype),
                        self.vae, dtype, gen,
                    )[0]
                elif cond_type == 'i2v_loop':
                    masks[i, 0, :, :, :] = 1
                    masks[i, -1, :, :, :] = 1
                    first_image = prepare_image_tensor(cond["first_image"], H, W)
                    last_image = prepare_image_tensor(cond["last_image"], H, W)
                    cond_latent[i] = vae_encode(
                        torch.concat([
                            first_image.cpu(),
                            torch.zeros(1, T - 2, 3, H, W),
                            last_image.cpu(),
                        ], dim=1).to(device, dtype=dtype),
                        self.vae, dtype, gen,
                    )[0]
            cond_latents = torch.cat((masks, cond_latent), dim=2)

        if num_additional_latents > 0:
            first_latent = cond_latents[:, : num_additional_latents, ...]
            cond_latents = torch.cat([first_latent, cond_latents], dim=1)
            lT = cond_latents.shape[1]

        cond_latents = cond_latents.unsqueeze(1).repeat(1, num_videos_per_prompt, 1, 1, 1, 1)
        cond_latents = cond_latents.view(batch_size * num_videos_per_prompt, lT, lC + 1, lH, lW)

        return latents, cond_latents, num_additional_latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        condition: Optional[List[Dict[str, Any]]] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        condition_cfg_type: Optional[str] = None,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The height in pixels of the generated image. This is set to 480 by default for the best results.
            width (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The width in pixels of the generated image. This is set to 720 by default for the best results.
            num_frames (`int`, defaults to `48`):
                Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
                contain 1 extra frame because CogVideoX is conditioned with (num_seconds * fps + 1) frames where
                num_seconds is 6 and fps is 8. However, since videos can be saved at any fps, the only condition that
                needs to be satisfied is that of divisibility mentioned above.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipelineOutput`] or `tuple`:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=self.transformer.config.max_text_seq_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            if pooled_prompt_embeds is not None:
                pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        latents, cond_latents, num_additional_latents = self.prepare_latents(
            batch_size=batch_size,
            num_videos_per_prompt=num_videos_per_prompt,
            num_channels_latents=self.transformer.config.in_channels,
            num_frames=num_frames,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
            condition=condition,
        )

        extra_set_timesteps_kwargs = {}

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps,
            **extra_set_timesteps_kwargs,
        )
        self._num_timesteps = len(timesteps)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        image_rotary_emb = (
            prepare_rotary_positional_embeddings(
                height,
                width,
                latents.size(1),
                vae_scale_factor_spatial=self.vae_scale_factor_spatial,
                patch_size=self.transformer.config.patch_size,
                patch_size_t=self.transformer.config.patch_size_t,
                attention_head_dim=self.transformer.config.attention_head_dim,
                device=device,
                sample_height=self.transformer.config.sample_height,
                sample_width=self.transformer.config.sample_width,
            )
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )


        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        if cond_latents is not None and do_classifier_free_guidance:
            if condition_cfg_type is not None and condition_cfg_type == 'joint':
                cond_latents = torch.cat([torch.zeros_like(cond_latents), cond_latents])
            else:
                cond_latents = torch.cat([cond_latents] * 2)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                timestep = t.expand(latent_model_input.shape[0])

                improved_kwargs = {}
                signatures = inspect.signature(self.transformer.forward).parameters
                if 'pooled_projections' in signatures:
                    improved_kwargs['pooled_projections'] = pooled_prompt_embeds

                if 'conditional_hidden_states' in signatures:
                    improved_kwargs['conditional_hidden_states'] = cond_latents

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                    **improved_kwargs
                )[0]
                noise_pred = noise_pred.float()

                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    if isinstance(self.scheduler, CogVideoXDPMImprovedScheduler):
                        extra_step_kwargs['timestep_prev'] = timesteps[i + 1] if i < len(timesteps) - 1 else -1
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        sample=latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents[:, num_additional_latents:]
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video)
