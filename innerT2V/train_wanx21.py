
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../')))


import math
import copy
import yaml
import json
import shutil
from pathlib import Path
from tqdm.auto import tqdm
from datetime import timedelta
import safetensors
import safetensors.torch
def load_file(filename, device = "cpu"):
    return safetensors.torch.load(open(filename, "rb").read())
safetensors.torch.load_file = load_file

import torch
import diffusers
import transformers
from transformers import AutoTokenizer, UMT5EncoderModel
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedType, init_empty_weights
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_density_for_timestep_sampling
from transformers import (
    AutoTokenizer,
)

from vae.wanx21_vae_add_logvar import AutoencoderKLWanImproved

from peft import LoraConfig

from args import get_args
from diffusers import WanTransformer3DModel
from text_encoder import wanx_compute_prompt_embeddings
from dataset.t2v_dataset_mask import T2VDataset
from dataset.sampler import DistributedBucketBatchSampler
from dataset.collate_mask import T2VCollateFunction
from dataset.efficient_data_meta_handler import EfficientDataMetaHandler
from dataset.utils import get_deterministic_worker_init_fn
from utils import (
    get_gradient_norm,
    get_optimizer,
    print_memory,
    reset_memory,
    unwrap_model,
    summarize_model_info,
    Timer,
)


import logging
from utils.logger import get_logger, add_handler, set_default_formatter
import torch.nn.functional as F


def main(args):
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.log_base, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, gradient_as_bucket_view=True)
    init_process_group_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=args.nccl_timeout))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
    )

    if accelerator.is_main_process:
        os.makedirs(logging_dir, mode=0o777, exist_ok=True)
    accelerator.wait_for_everyone()
    handler = logging.FileHandler((logging_dir / f"log-rank{accelerator.process_index}.txt").as_posix(), mode='a')
    add_handler(handler)
    set_default_formatter()
    logger = get_logger()

    logger.info(accelerator.state)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.bfloat16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    dataset_meta = EfficientDataMetaHandler(args.data_infos)

    buckets_config = None
    if getattr(args, 'buckets_config', None) is not None:
        import importlib
        buckets_config = importlib.import_module(args.buckets_config).buckets

    train_dataset = T2VDataset(
        dataset_meta=dataset_meta,
        resize_mode=args.resize_mode,
        height_buckets=args.height_buckets,
        width_buckets=args.width_buckets,
        frame_buckets=args.frame_buckets,
        min_step=args.min_step,
        max_step=args.max_step,
        candidate_caption_keys=getattr(args, 'caption_keys', None),
        candidate_caption_weights=getattr(args, 'caption_weights', None),
        frame_sampling_mode=getattr(args, 'frame_sampling_mode', 'interval'),
        buckets=buckets_config,
    )
    logger.info(f"[Config] Dataset loaded with number of samples = {len(train_dataset)}")

    conds_config = None

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.seed is not None:
        set_seed(args.seed, device_specific=True)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, mode=0o777, exist_ok=True)


    logger.info("[Process] Start loading pretrained models ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.wanx_pretrained_model_name_or_path,
        subfolder="tokenizer",
    )

    text_encoder = UMT5EncoderModel.from_pretrained(
        args.wanx_pretrained_model_name_or_path,
        subfolder="text_encoder",
    )

    load_dtype = torch.bfloat16
    model_path = args.wanx_pretrained_model_name_or_path
    if getattr(args, 'tuned_model_name_or_path', None) is not None:
        model_path = args.tuned_model_name_or_path
    transformer = WanTransformer3DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
    )
    ref_transformer = WanTransformer3DModel.from_pretrained(
        args.wanx_ref_model_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
    )

    vae = AutoencoderKLWanImproved.from_pretrained(
        args.wanx_pretrained_model_name_or_path,
        subfolder="vae",
    )

    scheduler = UniPCMultistepScheduler.from_pretrained(args.wanx_pretrained_model_name_or_path, subfolder="scheduler")

    logger.info('[Process] All pretrained model loaded')

    text_encoder.requires_grad_(False)

    vae.requires_grad_(False)

    logger.info("[Process] Start moving models to device ...")
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    logger.info(f"[Process] Models have been moved to device")
    ref_transformer.to(accelerator.device, dtype=weight_dtype)
    ref_lora_path = os.path.join(args.wanx_ref_model_name_or_path, "lora")
    if ref_lora_path is not None and os.path.exists(ref_lora_path):  
        logger.info(f"[Process] Reference transformer start loading LoRA with from {ref_lora_path} ...") 
        transformer_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights=True,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        ref_transformer.add_adapter(transformer_lora_config)
        try:
            ref_transformer.load_lora_adapter(ref_lora_path,
                use_safetensors=True,
                adapter_name='default',
                prefix=None,
            )
        except ValueError:
            ref_transformer.load_lora_adapter(ref_lora_path,
                use_safetensors=True,
                adapter_name='default',
            )
    ref_transformer.requires_grad_(False)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    transformer.requires_grad_(True)
    if args.lora_rank > 0:
        logger.info(f"[Process] Start adding LoRA with (rank={args.lora_rank}, alpha={args.lora_alpha}) ...")
        transformer.requires_grad_(False)
        transformer_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights=True,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        transformer.add_adapter(transformer_lora_config)
        if args.tuned_model_name_or_path is not None:
            lora_path = os.path.join(args.tuned_model_name_or_path, "lora")
        else:
            lora_path = None
        if lora_path is not None and os.path.exists(lora_path):
            logger.info(f"[Process] Transformer start loading LoRA with from {lora_path} ...") 
            try:
                transformer.load_lora_adapter(lora_path,
                    use_safetensors=True,
                    adapter_name='default',
                    prefix=None,
                )
            except ValueError:
                transformer.load_lora_adapter(lora_path,
                    use_safetensors=True,
                    adapter_name='default',
                )

    ref_lora_path = os.path.join(args.wanx_ref_model_name_or_path, "lora")
    if os.path.exists(ref_lora_path):  
        ref_transformer.add_adapter(transformer_lora_config)
        try:
            ref_transformer.load_lora_adapter(ref_lora_path,
                use_safetensors=True,
                adapter_name='default',
                prefix=None,
            )
        except ValueError:
            ref_transformer.load_lora_adapter(ref_lora_path,
                use_safetensors=True,
                adapter_name='default',
            )
    logger.info(f"[Config] Model info: {summarize_model_info(transformer)}")

    def save_model_hook(models, weights, output_dir):
        if not accelerator.is_main_process: return
        for model in models:
            if isinstance(unwrap_model(accelerator, model), type(unwrap_model(accelerator, transformer))):
                model = unwrap_model(accelerator, model)
                if args.lora_rank <= 0:
                    model.save_pretrained(
                        os.path.join(output_dir, "transformer"), safe_serialization=True, max_shard_size="5GB"
                    )
                else: 
                    if getattr(args, 'enable_model_parallel', False):
                        model.save_lora_adapter(os.path.join(output_dir, "lora"), safe_serialization=True, adapter_name='default')
                        model.unload_lora()
                        model.save_pretrained(
                            os.path.join(output_dir, "transformer"), safe_serialization=True, max_shard_size="5GB"
                        )
                        try:
                            model.load_lora_adapter(
                                os.path.join(output_dir, "lora"),
                                use_safetensors=True,
                                adapter_name='default',
                                prefix=None,
                            )
                        except ValueError:
                            model.load_lora_adapter(
                                os.path.join(output_dir, "lora"),
                                use_safetensors=True,
                                adapter_name='default',
                            )
                        except Exception as e:
                            raise ValueError(f"Error occurred during reload LoRA: {e}")       
                    else:
                        model.save_lora_adapter(os.path.join(output_dir, "lora"), safe_serialization=True, adapter_name='default')
                        base_model = copy.deepcopy(model)
                        base_model.unload_lora()
                        base_model.save_pretrained(
                            os.path.join(output_dir, "transformer"), safe_serialization=True, max_shard_size="5GB"
                        )
                        del base_model
            else:
                raise ValueError(f"Unexpected save model: {model.__class__}")

            if weights:
                weights.pop()

        scheduler.save_pretrained(os.path.join(output_dir, "scheduler"))

    def load_model_hook(models, input_dir):
        transformer_ = None
        init_under_meta = False

        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()

                if isinstance(unwrap_model(accelerator, model), type(unwrap_model(accelerator, transformer))):
                    transformer_ = unwrap_model(accelerator, model)
                else:
                    raise ValueError(f"Unexpected save model: {unwrap_model(accelerator, model).__class__}")
        else:
            with init_empty_weights():
                transformer_ = WanTransformer3DModel.from_config(
                    args.wanx_pretrained_model_name_or_path, subfolder="transformer"
                )
                init_under_meta = True

        if args.lora_rank > 0:
            transformer_.unload_lora()

        if os.path.exists(os.path.join(input_dir, "transformer")):
            load_model = WanTransformer3DModel.from_pretrained(os.path.join(input_dir, "transformer"))
            load_config = dict(copy.deepcopy(load_model.config))
            load_config.update(transformer_.config)
            transformer_.register_to_config(**load_config)
            transformer_.load_state_dict(load_model.state_dict(), assign=init_under_meta)
            del load_model

        if args.lora_rank > 0:
            try:
                transformer_.load_lora_adapter(os.path.join(input_dir, "lora"),
                    use_safetensors=True,
                    adapter_name='default',
                    prefix=None,
                )
            except ValueError:
                transformer_.load_lora_adapter(os.path.join(input_dir, "lora"),
                    use_safetensors=True,
                    adapter_name='default',
                )

        if args.mixed_precision == "fp16":
            cast_training_params([transformer_])

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)

    transformer_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    transformer_parameters_with_lr = {
        "params": transformer_parameters,
        "lr": args.learning_rate,
    }
    params_to_optimize = [transformer_parameters_with_lr]
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    optimizer = get_optimizer(
        params_to_optimize=params_to_optimize,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        beta3=args.beta3,
        epsilon=args.epsilon,
        weight_decay=args.weight_decay,
        prodigy_decouple=args.prodigy_decouple,
        prodigy_use_bias_correction=args.prodigy_use_bias_correction,
        prodigy_safeguard_warmup=args.prodigy_safeguard_warmup,
        use_8bit=args.use_8bit,
        use_4bit=args.use_4bit,
        use_torchao=args.use_torchao,
        use_deepspeed=use_deepspeed_optimizer,
        use_cpu_offload_optimizer=args.use_cpu_offload_optimizer,
        offload_gradients=args.offload_gradients,
    )

    if args.use_cpu_offload_optimizer:
        lr_scheduler = None
        logger.warning(
            "CPU Offload Optimizer cannot be used with DeepSpeed or builtin PyTorch LR Schedulers. If "
            "you are training with those settings, they will be ignored."
        )
    else:
        if use_deepspeed_scheduler:
            from accelerate.utils import DummyScheduler

            lr_scheduler = DummyScheduler(
                name=args.lr_scheduler,
                optimizer=optimizer,
                total_num_steps=args.max_train_steps * accelerator.num_processes,
                num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            )
        else:
            lr_scheduler = get_scheduler(
                args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                num_training_steps=args.max_train_steps * accelerator.num_processes,
                num_cycles=args.lr_num_cycles,
                power=args.lr_power,
            )

    logger.info("[Process] Start creating train dataloader ...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        batch_sampler=DistributedBucketBatchSampler(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=args.shuffle,
            drop_last=args.drop_last,
            generator=torch.Generator().manual_seed(args.seed),
        ),
        collate_fn=T2VCollateFunction(weight_dtype),
        num_workers=args.dataloader_num_workers,
        multiprocessing_context='fork',
        pin_memory=args.pin_memory,
        worker_init_fn=get_deterministic_worker_init_fn(args.seed),
    )
    accelerator.even_batches = False

    transformer, optimizer, lr_scheduler = accelerator.prepare(
        transformer, optimizer, lr_scheduler
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, 'args.yaml'), 'w') as f:
            yaml.dump(vars(args), f)

        tracker_name = args.tracker_name or "wanx-dpo"
        configs_for_track = vars(args)
        for key, value in configs_for_track.items():
            if isinstance(value, list) or isinstance(value, tuple) or isinstance(value, dict):
                configs_for_track[key] = str(value)
        accelerator.init_trackers(tracker_name, config=configs_for_track)

        reset_memory(accelerator.device)
        print_memory(accelerator.device, scope="before-training")

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info('\n'.join([
        "[Config] ***** Training Config (Partial) *****",
        f"  Num trainable parameters = {num_trainable_parameters}",
        f"  Num examples = {len(train_dataset)}",
        f"  Num batches each epoch = {len(train_dataloader)}",
        f"  Num epochs = {args.num_train_epochs}",
        f"  Instantaneous batch size per device = {args.train_batch_size}",
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}",
        f"  Gradient accumulation steps = {args.gradient_accumulation_steps}",
        f"  Total optimization steps = {args.max_train_steps}",
    ]))
    global_step = 0
    first_epoch = 0

    if getattr(args, 'load_from_checkpoint', None) is not None:
        load_path = getattr(args, 'load_from_checkpoint')
        logger.info(f"[Process] Loading checkpoint from {load_path}")
        accelerator.load_state(load_path)

    if not args.resume_from_checkpoint:
        initial_global_step = 0
    else:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logger.warning(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            logger.info(f"[Process] Resuming training from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))

            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    logger.info(f"[Config] Final Model Info: {summarize_model_info(transformer)}")
    logger.info(f"[Config] Final Ref Model Info: {summarize_model_info(ref_transformer)}")
    logger.info(f"[Config] Final Model Config: {model_config}")

    timer = Timer(10)

    yita_range = 0.15

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        counter = 0
        train_loss = 0.0
        implicit_acc_accumulated = 0.0
        implicit_acc_accumulated_mask = 0.0
        
        for step, batch in enumerate(train_dataloader):
            counter += 1
            models_to_accumulate = [transformer]

            with accelerator.accumulate(models_to_accumulate), timer("iteration"):
                pos_videos = batch["pos_videos"].to(accelerator.device, non_blocking=True)
                neg_videos = batch["neg_videos"].to(accelerator.device, non_blocking=True)
                masks = batch['masks'].to(accelerator.device, non_blocking=True)
                masks_shape = masks.shape[0] * masks.shape[1] * masks.shape[3] * masks.shape[4]
                assert masks.shape[2] == 1
                masks = masks.permute(0, 2, 1, 3, 4)
                prompts = batch["prompts"]
                yitas = batch['yitas'].to(accelerator.device, non_blocking=True)
                yitas = yitas.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                yitas = 1 + (yitas - 0.75) / yita_range

                videos = torch.cat([pos_videos, neg_videos])
                    
                timer.tic("vae-encode")
                latent_dist = vae.encode(videos.permute(0, 2, 1, 3, 4)).latent_dist

                latent = latent_dist.sample()
                latent = latent.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                model_input = latent
                timer.toc()

                timer.tic("text-encode")

                prompt_embeds = wanx_compute_prompt_embeddings(
                    tokenizer,
                    text_encoder,
                    prompts,
                    512,
                    accelerator.device,
                    weight_dtype,
                    requires_grad=False,
                )
                prompt_embeds = prompt_embeds.repeat(2, 1, 1)
                timer.toc()
              
                timer.tic("prepare-latent")
                noise = torch.randn_like(model_input)
                batch_size, num_frames, num_channels, height, width = model_input.shape

                u = compute_density_for_timestep_sampling(
                    weighting_scheme='logit_normal',
                    batch_size=batch_size,
                    logit_mean=0,
                    logit_std=1,
                    device=model_input.device,
                )
                timesteps = (u * scheduler.config.num_train_timesteps).long()

                noise = noise.chunk(2)[0].repeat(2, 1, 1, 1, 1)
                timesteps_half = timesteps.chunk(2)[0]


                timesteps = timesteps_half.repeat(2)

                noisy_model_input = scheduler.add_noise(model_input, noise, timesteps)
                timer.toc()
                  
                timer.tic("forward")
                model_output = transformer(
                    hidden_states=noisy_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timesteps,
                    attention_kwargs=None,
                    return_dict=False
                )[0]
                model_pred = model_output

                target = noise - model_input
                model_losses = (model_pred - target).pow(2)
                model_losses_w, model_losses_l = model_losses.chunk(2)

                model_losses_w_mask = model_losses_w * masks * yitas * masks_shape / torch.sum(masks)
                model_losses_l_mask = model_losses_l * masks * yitas * masks_shape / torch.sum(masks)
                model_losses_w = model_losses_w.mean(dim=[1,2,3,4])
                model_losses_l = model_losses_l.mean(dim=[1,2,3,4])
                model_losses_w_mask = model_losses_w_mask.mean(dim=[1,2,3,4])
                model_losses_l_mask = model_losses_l_mask.mean(dim=[1,2,3,4])
                raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())
                raw_model_loss_mask = 0.5 * (model_losses_w_mask.mean() + model_losses_l_mask.mean())
                model_diff = model_losses_w - model_losses_l
                model_diff_mask = model_losses_w_mask - model_losses_l_mask

            
                with torch.no_grad():
                    ref_model_output = ref_transformer(
                        hidden_states=noisy_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timesteps,
                        attention_kwargs=None,
                        return_dict=False,
                    )[0].detach()

                ref_model_pred = ref_model_output
                ref_losses = (ref_model_pred - target).pow(2)
                ref_losses_w, ref_losses_l = ref_losses.chunk(2)
                ref_losses_w_mask = ref_losses_w * masks * yitas * masks_shape / torch.sum(masks)
                ref_losses_l_mask = ref_losses_l * masks * yitas * masks_shape / torch.sum(masks)
                ref_losses_w = ref_losses_w.mean(dim=[1,2,3,4])
                ref_losses_l = ref_losses_l.mean(dim=[1,2,3,4])
                ref_losses_w_mask = ref_losses_w_mask.mean(dim=[1,2,3,4])
                ref_losses_l_mask = ref_losses_l_mask.mean(dim=[1,2,3,4])
                ref_diff = ref_losses_w - ref_losses_l
                ref_diff_mask = ref_losses_w_mask - ref_losses_l_mask
                raw_ref_loss = 0.5 * (ref_losses_w.mean() + ref_losses_l.mean())
                raw_ref_loss_mask = 0.5 * (ref_losses_w_mask.mean() + ref_losses_l_mask.mean())

                scale_term = -0.5 * args.beta_dpo

                weights = 1

                scale_term *= weights

                inside_term = scale_term * (model_diff - ref_diff)
                inside_term_mask = scale_term * (model_diff_mask - ref_diff_mask)

                implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
                implicit_acc_mask = (inside_term_mask > 0).sum().float() / inside_term_mask.size(0)
                assert implicit_acc <= 1, f'Invalid implicit acc: {implicit_acc}, which should between 0~1.'
                assert implicit_acc_mask <= 1, f'Invalid implicit acc: {implicit_acc_mask}, which should between 0~1.'

                loss = -1 * F.logsigmoid(inside_term)
                loss_dpo_mask = -1 * F.logsigmoid(inside_term_mask)

                loss_sft = weights * model_losses_w
                loss_sft_mask = weights * model_losses_w_mask

                loss = (args.dpo_lambda * loss + args.mask_dpo_lambda * loss_dpo_mask + args.sft_lambda_mask * loss_sft_mask + args.sft_lambda * loss_sft).mean()

                avg_loss = accelerator.gather(loss.repeat(batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                avg_model_mse = accelerator.gather(raw_model_loss.repeat(batch_size)).mean().item()
                avg_model_mse_mask = accelerator.gather(raw_model_loss_mask.repeat(batch_size)).mean().item()
                avg_ref_mse = accelerator.gather(raw_ref_loss.repeat(batch_size)).mean().item()
                avg_ref_mse_mask = accelerator.gather(raw_ref_loss_mask.repeat(batch_size)).mean().item()
                avg_acc = accelerator.gather(implicit_acc).mean().item()
                avg_acc_mask = accelerator.gather(implicit_acc_mask).mean().item()
                implicit_acc_accumulated += avg_acc / args.gradient_accumulation_steps
                implicit_acc_accumulated_mask += avg_acc_mask / args.gradient_accumulation_steps
                
                timer.toc()
                print_memory(accelerator.device, scope="before-backward")
                timer.tic("backward")
                accelerator.backward(loss)

                gradient_norm_before_clip = None
                gradient_norm_after_clip = None
                if accelerator.sync_gradients:
                    gradient_norm_before_clip = get_gradient_norm(transformer.parameters())
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                    gradient_norm_after_clip = get_gradient_norm(transformer.parameters())

                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()
                    optimizer.zero_grad()

                if not args.use_cpu_offload_optimizer:
                    lr_scheduler.step()
                reset_memory(accelerator.device)
                timer.toc()

                timer.tic("log-and-checkpoint")
                if accelerator.sync_gradients:
                    global_step += 1

                    if global_step % args.checkpointing_steps == 0:
                        if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                            if args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                    accelerator.wait_for_everyone()
    
                    last_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else args.learning_rate
                    logs = {
                        "loss": accelerator.reduce(loss.detach().clone(), reduction="mean").item(),
                        "lr": last_lr,
                        'train_loss': train_loss,
                        'model_mse_unaccumulated': avg_model_mse,
                        'model_mse_mask_unaccumulated': avg_model_mse_mask,
                        'ref_mse_unaccumulated': avg_ref_mse,
                        'ref_mse_mask_unaccumulated': avg_ref_mse_mask,
                        'implicit_acc_accumulated': implicit_acc_accumulated,
                        'implicit_acc':avg_acc,
                        'implicit_acc_accumulated_mask': implicit_acc_accumulated_mask,
                        'implicit_acc_mask': avg_acc_mask,
                    }
                    train_loss = 0.0
                    implicit_acc_accumulated = 0.0
                    if accelerator.distributed_type != DistributedType.DEEPSPEED:
                        logs.update(
                            {
                                "gradnorm_before_clip": gradient_norm_before_clip,
                                "gradnorm_after_clip": gradient_norm_after_clip,
                            }
                        )
                    progress_bar.set_postfix(**logs, refresh=False)
                    progress_bar.update(1)
                    accelerator.log(logs, step=global_step)
                timer.toc()

            if global_step % 10 == 0:
                print_memory(accelerator.device, scope="after-iteration")

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    args = get_args()
    main(args)
