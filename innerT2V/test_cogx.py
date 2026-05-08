import os
import sys

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../')))

import json
import torch
import shutil
import argparse
from tqdm import tqdm
import safetensors
import safetensors.torch
def load_file(filename, device = "cpu"):
    return safetensors.torch.load(open(filename, "rb").read())
safetensors.torch.load_file = load_file

from accelerate import Accelerator
from diffusers.utils import export_to_video
from transformers import (
    T5EncoderModel,
)
from diffusers import AutoencoderKLCogVideoX

from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from pipeline.pipeline_cogvideox_improved import CogVideoXImprovedPipeline
from utils import summarize_model_info

import logging
from utils.logger import get_logger, add_handler, set_default_formatter


VALID_TUNED_MODULES = {
    'text_encoder': T5EncoderModel,
    'transformer': CogVideoXTransformer3DModel,
    'vae': AutoencoderKLCogVideoX,
}

SYSTEM_POSITIVE_PROMPTS = "highly detailed, perfect without deformations, ultra HD, "
SYSTEM_NEGATIVE_PROMPTS = "blurring, dirty, messy, low quality, cartoon, drawing, anime"

def prepare_prompt(
    info: dict,
    key: str,
    add_pos_prompt: bool = False,
    add_neg_prompt: bool = False,
):
    prompt = info.get(key or 'original', None) or info['original']
    prompt = prompt.strip()
    if add_pos_prompt:
        prompt = f'{SYSTEM_POSITIVE_PROMPTS}{prompt}'

    neg_prompt = None
    if add_neg_prompt:
        neg_prompt = SYSTEM_NEGATIVE_PROMPTS

    return prompt, neg_prompt


def main(cml_args=None, only_get_args: bool = False):
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompts", type=str, nargs='+', default=None, help="prompts for the video generation")
    parser.add_argument("--prompts_dict", type=str, default=None, help="prompts formatted in json-string")
    parser.add_argument("--prompts_file", type=str, default=None, help="Text file containing prompts for the video generation")

    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the generated video")
    parser.add_argument("--log_path", type=str, default=None, help="Log file path. If not set, use stdout")

    parser.add_argument("--base_modules_dir", type=str, default="THUDM/CogVideoX1.5-5B")
    parser.add_argument("--tuned_modules_dir", type=str, default=None)
    parser.add_argument("--lora_scale", type=float, default=1.0, help="Lora scale")

    parser.add_argument("--fps", type=int, default=24, help="Frames per second for the output video")
    parser.add_argument('--height', type=int, default=768)
    parser.add_argument('--width', type=int, default=1360)
    parser.add_argument('--sample_frames', type=int, default=8)
    parser.add_argument('--num_videos_per_prompt', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--add_pos_prompt', action='store_true', default=False)
    parser.add_argument('--add_neg_prompt', action='store_true', default=False)

    args, _ = parser.parse_known_args(cml_args)

    if args.log_path is not None:
        handler = logging.FileHandler(args.log_path, mode='a')
        add_handler(handler)
    set_default_formatter()
    logger = get_logger()

    prompts = []
    if args.prompts_file is not None:
        with open(args.prompts_file, 'r') as f:
            prompts = json.load(f)
    if args.prompts is not None:
        custom_prompts = ' '.join(args.prompts).split(":::")
        prompts.extend([{'original': p} for p in custom_prompts])
    if args.prompts_dict is not None:
        parsed_data = json.loads(args.prompts_dict)
        if not isinstance(parsed_data, list):
            parsed_data = [parsed_data]
        prompts.extend(parsed_data)
    args.all_prompts = prompts

    if only_get_args:
        return args, parser

    accelerator = Accelerator()

    tuned_modules = {}
    has_lora = False

    if args.tuned_modules_dir is not None and os.path.isdir(args.tuned_modules_dir):
        for subdir in os.listdir(args.tuned_modules_dir):
            if subdir == 'lora': has_lora = True
            if subdir not in VALID_TUNED_MODULES: continue
            logger.info(f"Loading tuned module {subdir} from {args.tuned_modules_dir}")
            module = VALID_TUNED_MODULES[subdir].from_pretrained(
                args.tuned_modules_dir, subfolder=subdir,
                low_cpu_mem_usage=True,
            )
            tuned_modules[subdir] = module

    pipe = CogVideoXImprovedPipeline.from_pretrained(
        args.base_modules_dir,
        **tuned_modules,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    print(f'has lora is: {has_lora}')

    if has_lora:
        lora_scale = getattr(args, "lora_scale", 1.0)
        logger.info(f"Loading Lora with scale {lora_scale}")
        try:
            pipe.transformer.load_lora_adapter(os.path.join(args.tuned_modules_dir, "lora"),
                use_safetensors=True,
                adapter_name='default',
                prefix=None,
            )
        except ValueError:
            pipe.transformer.load_lora_adapter(os.path.join(args.tuned_modules_dir, "lora"),
                use_safetensors=True,
                adapter_name='default',
            )
        pipe.transformer.fuse_lora(lora_scale=1.0, adapter_names=['default'])
        pipe.transformer.unload_lora()

    pipe.transformer.to(dtype=torch.bfloat16)
    pipe.text_encoder.to(dtype=torch.bfloat16)
    pipe.vae.to(dtype=torch.bfloat16)

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    n_chunks = accelerator.num_processes
    prompts = [prompts[i::n_chunks] for i in range(n_chunks)][accelerator.process_index]
    logger.info(f"Assign {len(prompts)} prompts to rank {accelerator.process_index}.")

    pipe.enable_model_cpu_offload(device=accelerator.device)

    subdir = []
    if has_lora:
        subdir.append(f'lora{lora_scale:.1f}')
    subdir = '-'.join(subdir)
    args.output_dir = os.path.join(args.output_dir, subdir)

    expander_key = 'long'

    os.makedirs(args.output_dir, exist_ok=True)
    for info in tqdm(prompts):
        info.update({'fps': args.fps})
        prompt, negative_prompt = prepare_prompt(
            info,
            key=expander_key,
            add_pos_prompt=args.add_pos_prompt,
            add_neg_prompt=args.add_neg_prompt
        )

        file_basename = prompt[len(SYSTEM_POSITIVE_PROMPTS):] if args.add_pos_prompt else prompt
        if len(file_basename) > 200:
            file_basename = file_basename[:200]
            logger.warning(f"File name is too long, truncated to 200 characters.")
        is_absent = False
        for i in range(args.num_videos_per_prompt):
            output_file = os.path.join(args.output_dir, f"{file_basename}-{i}.mp4")
            if not os.path.exists(output_file):
                is_absent = True
        is_absent = is_absent and not os.path.exists(os.path.join(args.output_dir, f"{file_basename}.mp4"))
        if not is_absent:
            logger.info(f"Skip {prompt}")
            continue

        logger.info(f"Prompt is: {prompt}")
        logger.info(f"Negative prompt is: {negative_prompt}")

        videos = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=args.num_videos_per_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.sample_frames,
            use_dynamic_cfg=True,
            guidance_scale=6.0,
            generator=torch.Generator().manual_seed(args.seed),
        ).frames

        for i, video in enumerate(videos):
            output_file = os.path.join(args.output_dir, f"{file_basename}-{i}.mp4")
            if args.num_videos_per_prompt == 1:
                output_file = os.path.join(args.output_dir, f"{file_basename}.mp4")
            export_to_video(video, output_file, fps=args.fps)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--launch_mode", type=str, default="main",
                            choices=["main"], help="Launch mode")
        args, extra_args = parser.parse_known_args()
        main(extra_args)
    except Exception as e:
        logger = get_logger()
        logger.exception(e)
        exit(1)
    exit(0)