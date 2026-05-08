import argparse


def _get_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--wanx_pretrained_model_name_or_path",
        type=str,
        default=None,
        help='Path to wanx2.1 pretrained model (for loading wanx2.1 VAE)'
    )
    parser.add_argument(
        "--ref_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--wanx_ref_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tuned_model_name_or_path",
        type=str,
        default=None,
        help="Path to the finetuned model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        '--max_text_seq_length',
        type=int,
        default=226,
    )
    parser.add_argument(
        '--beta_dpo',
        type=float,
        default=5000,
        help='the weight of dpo loss'
    )
    parser.add_argument(
        '--sft_lambda',
        type=float,
        default=0.1,
        help='the weight of sft loss'
    )
    parser.add_argument(
        '--dpo_lambda',
        type=float,
        default=1.0,
        help='the weight of pure dpo loss'
    )
    parser.add_argument(
        '--mask_dpo_lambda',
        type=float,
        default=1.0,
        help='the weight of masked dpo loss'
    )
    parser.add_argument(
        '--sft_lambda_mask',
        type=float,
        default=0.1,
        help='the weight of sft loss in masked regions'
    )


def _get_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--height_buckets",
        nargs="+",
        type=int,
        default=[768],
    )
    parser.add_argument(
        "--width_buckets",
        nargs="+",
        type=int,
        default=[1360],
    )
    parser.add_argument(
        "--frame_buckets",
        nargs="+",
        type=int,
        default=[49],
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Whether or not to use the pinned memory setting in pytorch dataloader.",
    )
    parser.add_argument(
        '--shuffle',
        action='store_true',
        default=False,
        help='If activated, shuffle this dataset'
    )
    parser.add_argument(
        '--drop_last',
        action='store_true',
        default=False,
        help='If activated, drop the final data item away'
    )
    parser.add_argument(
        '--data_infos',
        nargs='+',
        type=str,
        default=None,
        help='The json file that contains the information of a dataset.'
    )
    parser.add_argument(
        '--min_step', type=int, default=1,
        help='The minimal step for sampling frames'
    )
    parser.add_argument(
        '--max_step', type=int, default=8,
        help='The maximum step for sampling frames'
    )
    parser.add_argument(
        '--caption_keys',
        nargs='+',
        type=str,
        default=None,
        help='The keys of the caption'
    )
    parser.add_argument(
        '--caption_weights',
        nargs='+',
        type=float,
        default=None,
        help='The weights of the caption'
    )
    parser.add_argument(
        "--resize_mode",
        type=str,
        default='center',
        choices=['center', 'random', 'none'],
        help="All input videos are cropped in this mode.",
    )
    parser.add_argument(
        '--frame_sampling_mode',
        type=str,
        default='interval',
        choices=('uniform', 'interval', 'continuous'),
        help='The mode of sampling frames'
    )
    parser.add_argument(
        '--buckets_config',
        type=str,
        default=None,
        choices=('configs.buckets.256px_mf', 'configs.buckets.256px_25f', 'configs.buckets.480px_mf', 'configs.buckets.768px_mf', 'configs.buckets.480px_mf_mask'),
        help='The buckets config file'
    )
    parser.add_argument(
        "--batch_size_scales",
        type=int,
        nargs='+',
        default=None,
        help=(
            'The batch size scales for different nframes, '
            'e.g., when passing 1 4 121 1, a dictionary of {1: 4, 121: 1} will be parsed, '
            'and we willscale the batch size by 4 for 1 frame, by 1 for 121 frames, respectively'
        ),
    )


def _get_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--lora_rank", type=int, default=0, help="The rank for LoRA matrices.")
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=0,
        help="The lora_alpha to compute scaling factor (lora_alpha / rank) for LoRA matrices.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU. "
            "Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command. Use this "
            "argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cogvideox-sft",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        '--log_base',
        type=str, 
        default='cogvideox-sft'
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides `--num_train_epochs`.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--load_from_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--enable_slicing",
        action="store_true",
        default=False,
        help="Whether or not to use VAE slicing for saving memory.",
    )
    parser.add_argument(
        "--enable_tiling",
        action="store_true",
        default=False,
        help="Whether or not to use VAE tiling for saving memory.",
    )

    parser.add_argument(
        "--enable_model_parallel",
        action="store_true",
        default=False,
        help="Whether or not to use model parallelism for training."
    )


def _get_optimizer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--optimizer",
        type=lambda s: s.lower(),
        default="adam",
        choices=["adam", "adamw", "prodigy", "came"],
        help=("The optimizer type to use."),
    )
    parser.add_argument(
        "--use_8bit",
        action="store_true",
        help="Whether or not to use 8-bit optimizers from `bitsandbytes` or `bitsandbytes`.",
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Whether or not to use 4-bit optimizers from `torchao`.",
    )
    parser.add_argument(
        "--use_torchao", action="store_true", help="Whether or not to use the `torchao` backend for optimizers."
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.95,
        help="The beta2 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--beta3",
        type=float,
        default=None,
        help="Coefficients for computing the Prodigy optimizer's stepsize using running averages. If set to None, uses the value of square root of beta2.",
    )
    parser.add_argument(
        "--prodigy_decouple",
        action="store_true",
        help="Use AdamW style decoupled weight decay.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-04,
        help="Weight decay to use for optimizer.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--prodigy_use_bias_correction",
        action="store_true",
        help="Turn on Adam's bias correction.",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        action="store_true",
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage.",
    )
    parser.add_argument(
        "--use_cpu_offload_optimizer",
        action="store_true",
        help="Whether or not to use the CPUOffloadOptimizer from TorchAO to perform optimization step and maintain parameters on the CPU.",
    )
    parser.add_argument(
        "--offload_gradients",
        action="store_true",
        help="Whether or not to offload the gradients to CPU when using the CPUOffloadOptimizer from TorchAO.",
    )


def _get_configuration_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tracker_name", type=str, default=None, help="Project tracker name")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Directory where logs are stored.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--nccl_timeout",
        type=int,
        default=7200,
        help="Maximum timeout duration before which allgather, or related, operations fail in multi-GPU/multi-node training settings.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )


def get_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for CogVideoX.")

    _get_model_args(parser)
    _get_dataset_args(parser)
    _get_training_args(parser)
    _get_optimizer_args(parser)
    _get_configuration_args(parser)

    args = parser.parse_args()

    if args.batch_size_scales is not None:
        i = iter(args.batch_size_scales)
        args.batch_size_scales = dict(zip(i, i))

    return args
