import torch
import numpy as np
from typing import List, Optional, Tuple, Union

from diffusers.schedulers.scheduling_dpm_cogvideox import (
    DDIMSchedulerOutput, CogVideoXDPMScheduler, register_to_config, randn_tensor
)


class CogVideoXDPMImprovedScheduler(CogVideoXDPMScheduler):

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
        beta_schedule: str = "scaled_linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        timestep_spacing: str = "leading",
        rescale_betas_zero_snr: bool = False,
        snr_shift_scale: float = 3.0,
        shift_alpha: float = 1.0,
    ):
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            trained_betas=trained_betas,
            clip_sample=clip_sample,
            set_alpha_to_one=set_alpha_to_one,
            steps_offset=steps_offset,
            prediction_type=prediction_type,
            clip_sample_range=clip_sample_range,
            sample_max_value=sample_max_value,
            timestep_spacing=timestep_spacing,
            rescale_betas_zero_snr=rescale_betas_zero_snr,
            snr_shift_scale=snr_shift_scale,
        )

        self._full_timesteps_before_shift = self.timesteps.clone()
        self._timesteps_before_shift = self.timesteps.clone()

    def step(
        self,
        model_output: torch.Tensor,
        old_pred_original_sample: torch.Tensor,
        timestep: int,
        timestep_back: int,
        timestep_prev: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )



        timestep_prev = timestep_prev

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[timestep_prev] if timestep_prev >= 0 else self.final_alpha_cumprod
        alpha_prod_t_back = self.alphas_cumprod[timestep_back] if timestep_back is not None else None

        beta_prod_t = 1 - alpha_prod_t

        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        h, r, lamb, lamb_next = self.get_variables(alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back)
        mult = list(self.get_mult(h, r, alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back))
        mult_noise = (1 - alpha_prod_t_prev) ** 0.5 * (1 - (-2 * h).exp()) ** 0.5

        noise = randn_tensor(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
        prev_sample = mult[0] * sample - mult[1] * pred_original_sample + mult_noise * noise

        if old_pred_original_sample is None or timestep_prev < 0:
            return prev_sample, pred_original_sample
        else:
            denoised_d = mult[2] * pred_original_sample - mult[3] * old_pred_original_sample
            noise = randn_tensor(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            x_advanced = mult[0] * sample - mult[1] * denoised_d + mult_noise * noise

            prev_sample = x_advanced

        if not return_dict:
            return (prev_sample, pred_original_sample)

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
