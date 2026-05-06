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
from typing import Optional


import safetensors
import safetensors.torch
def load_file(filename, device = "cpu"):
    return safetensors.torch.load(open(filename, "rb").read())
safetensors.torch.load_file = load_file

from accelerate import Accelerator
from diffusers.utils import export_to_video
from transformers import (
    AutoTokenizer,
    T5EncoderModel,
    CLIPTextModel,
    LlamaModel,
)
from diffusers import AutoencoderKLCogVideoX

from acceleration.attention import optimize_transformer
from acceleration.distributed.config import get_parallel_config
from acceleration.distributed import hybrid_parallelize_pipeline
from extensions.xfuser.core.distributed import (
    init_distributed_environment,
    get_data_parallel_world_size,
    get_data_parallel_rank,
    is_dp_last_group,
)

from transformer.cogvideox_improved_transformer_3d import CogVideoXImprovedTransformer3DModel
from pipeline.pipeline_cogvideox_improved import CogVideoXImprovedPipeline
from utils import summarize_model_info, PROMPT_EXPANDER
from utils.io import load_image

import logging
from utils.logger import get_logger, add_handler, set_default_formatter


VALID_TUNED_MODULES = {
    'text_encoder': T5EncoderModel,
    'transformer': CogVideoXImprovedTransformer3DModel,
    'vae': AutoencoderKLCogVideoX,
}

SYSTEM_POSITIVE_PROMPTS = "highly detailed, perfect without deformations, ultra HD, "
SYSTEM_NEGATIVE_PROMPTS = "blurring, dirty, messy, low quality, cartoon, drawing, anime"

def prepare_prompt(
    info: dict,
    key: str,
    expander,
    add_pos_prompt: bool = False,
    add_neg_prompt: bool = False,
):
    prompt = info.get(key or 'original', None) or info['original']
    if expander is not None:
        prompt = expander(prompt, mode='t2v-with-examples')
    prompt = prompt.strip()
    if add_pos_prompt:
        prompt = f'{SYSTEM_POSITIVE_PROMPTS}{prompt}'


    neg_prompt = None
    if add_neg_prompt:
        neg_prompt = SYSTEM_NEGATIVE_PROMPTS

    return prompt, neg_prompt


def prepare_motion_score(
    prompt: str,
    info: dict,
    expander,
    expander_key,
    base_score: float,
):
    logger = get_logger()
    vmaf_score = base_score
    if vmaf_score > 100:
        if 'dynamic_degree' in info.get('dimension', []):
            vmaf_score = 4.0 + (vmaf_score % 101)
            logger.info(f'[Auto-VMafMotion-Injection] testing dynamic_degree dimension, set vmaf_score to {vmaf_score}')
        else:
            vmaf_score = vmaf_score // 100 - 1
            logger.info(f'[Auto-VMafMotion-Injection] testing other dimensions, set vmaf_score to {vmaf_score}')
    elif vmaf_score < 0:
        score_key = f'motion-score-{expander_key}'
        if score_key in info:
            vmaf_score = info[score_key]
        else:
            if expander is None:
                raise ValueError("You must set expander if vmaf_score < 0")
            vmaf_score_str = expander(prompt, mode='motion-score-with-examples')
            try:
                vmaf_score = float(vmaf_score_str)
            except:
                logger.warning(f"[Auto-VMafMotion-Injection] failed to parse vmaf_score for prompt {prompt}, get following response: {vmaf_score_str}")
                vmaf_score = 0.0
    return vmaf_score


def prepare_condition_image(path):

    if path is None:
        return None

    image = load_image(path)
    tensor = torch.from_numpy(image).permute(2, 0, 1) \
                .float().div(255.).clip(0, 1).mul(2).sub(1)
    return tensor


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
    parser.add_argument(
        '--condition_cfg_type',
        type=str,
        default=None,
        choices=('joint'),
        help=(
            'default: m(lat, 0, 0) + g * [m(lat, text, cond) - m(lat, 0, cond)];\n'
            'joint: m(lat, 0, 0) + g * [m(lat, text, cond) - m(lat, 0, 0)]'
        ),
    )

    parser.add_argument(
        '--prompt_expander', '--enhance_prompt',
        type=str,
        default=None,
        help=(
            "Prompt expander. "
            "Four types of expanders can be specified.\n"
            "1. `short`: equal to unset, which means no prompt expansion;\n"
            "2. offline expanders such as `gpt4o`, `deepseek_v1`, `gpt4o_wo_scenecut_v2`: make sure all prompt dicts contain the corresponding key;\n"
            "3. `{KEY}?API?{MODEL_NAME}`: such as `GPT4o-V1?API?GPT4o`, `DS-V1?API?DeepSeek`; the {KEY} field to used to record the extended prompts in output file;\n"
            "4. `{KEY}?Local?{MODEL_DIR}`: such as `Qwen-V1?Local?/path/to/Qwen2.5-14B-Instruct`; the {KEY} field to used to record the extended prompts in output file"
        )
    )
    parser.add_argument('--add_pos_prompt', action='store_true', default=False)
    parser.add_argument('--add_neg_prompt', action='store_true', default=False)

    parser.add_argument('--enable_parallel', action='store_true')

    args, _ = parser.parse_known_args(cml_args)

    if args.log_path is not None:
        handler = logging.FileHandler(args.log_path, mode='a')
        add_handler(handler)
    set_default_formatter()
    logger = get_logger()

 
    if args.prompt_expander in ['gpt4o', 'deepseek_v1', 'gpt4o_wo_scenecut_v2']:
        args.add_pos_prompt = True
        args.add_neg_prompt = True
    if args.prompt_expander == 'short':
        args.prompt_expander = None


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
    if args.enable_parallel:
        init_distributed_environment()
        torch.cuda.set_device(accelerator.device)

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

    logger.info(f"Model Info: {summarize_model_info(pipe.transformer)}")

    if args.enable_parallel:
        parallel_config = get_parallel_config('test')
        pipe = hybrid_parallelize_pipeline(pipe, **parallel_config)
    else:
        logger.info("accelerate transformer using sage")
        pipe.transformer = optimize_transformer(pipe.transformer, 'sage')

    if args.enable_parallel:
        if parallel_config['data_parallel_degree'] > 1:
            n_chunks = get_data_parallel_world_size()
            prompts = [prompts[i::n_chunks] for i in range(n_chunks)][get_data_parallel_rank()]
    else:
        n_chunks = accelerator.num_processes
        prompts = [prompts[i::n_chunks] for i in range(n_chunks)][accelerator.process_index]
    logger.info(f"Assign {len(prompts)} prompts to rank {accelerator.process_index}.")

    pipe.enable_model_cpu_offload(device=accelerator.device)

    subdir = []
    if has_lora:
        subdir.append(f'lora{lora_scale:.1f}')


    if args.condition_cfg_type is not None:
        subdir.append(f'cfg#{args.condition_cfg_type}')
    subdir = '-'.join(subdir)
    args.output_dir = os.path.join(args.output_dir, subdir)

    expander_key = None
    prompt_expander = None
    if args.prompt_expander is not None:
        expander_key = args.prompt_expander.split('?')[0]
        for info in prompts:
            if expander_key not in info:
                logger.info(f"Cannot find `{expander_key}` in testing meta data. Try creating PromptExpander ...")
                if len(expander_key.split('?')) != 3:
                    raise ValueError("prompt_expander must be in format `{KEY}?[API|Local]?{MODEL_NAME}` when {KEY} is not in testing meta data")
                _, expander_type, expander_name  = args.prompt_expander.split('?')
                prompt_expander = PROMPT_EXPANDER.get(expander_type)(expander_name)
                break

    os.makedirs(args.output_dir, exist_ok=True)
    for info in tqdm(prompts):
        info.update({'fps': args.fps})
        prompt, negative_prompt = prepare_prompt(
            info,
            key=expander_key,
            expander=prompt_expander,
            add_pos_prompt=args.add_pos_prompt,
            add_neg_prompt=args.add_neg_prompt,
    
        )

        vmaf_score = None
        if args.vmaf_score is not None:
            vmaf_score = prepare_motion_score(
                prompt,
                info,
                prompt_expander,
                expander_key,
                args.vmaf_score,
            )

        file_basename = info.get('id', info['original'])
        if len(file_basename) > 100:
            file_basename = file_basename[:100]
            logger.warning(f"File name is too long, truncated to 500 characters.")
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
        logger.info(f"MetricCond injection: aes-{args.aes_score}, clarity-{args.clarify_score}, vmaf-{vmaf_score}")

        condition_info = None
        if 'cond_type' in info:
            cond_type = info['cond_type']
            condition_info = {
                'cond_type': cond_type,
                'first_image': prepare_condition_image(info.get('first_image_file', None)),
                'last_image': prepare_condition_image(info.get('last_image_file', None)),
            }
            logger.info(f"Condition type is: {cond_type}")

        videos = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=args.num_videos_per_prompt,
            condition=condition_info,
            height=args.height,
            width=args.width,
            num_frames=args.sample_frames,
            use_dynamic_cfg=True,
            guidance_scale=6.0,
            generator=torch.Generator().manual_seed(args.seed),
            condition_cfg_type=getattr(args, 'condition_cfg_type', None),
        ).frames
        if (args.enable_parallel and is_dp_last_group()) or not args.enable_parallel:

            for i, video in enumerate(videos):
                output_file = os.path.join(args.output_dir, f"{file_basename}-{i}.mp4")
                if args.num_videos_per_prompt == 1:
                    output_file = os.path.join(args.output_dir, f"{file_basename}.mp4")
            
                tmp_file = f'/tmp/{os.path.basename(output_file)}'
                export_to_video(video, tmp_file, fps=args.fps)
                shutil.move(tmp_file, output_file)



def unparse_args(parser, namespace):
    arguments = []
    for action in parser._actions:
        dest = action.dest
        if dest == argparse.SUPPRESS or dest is None:
            continue
        value = getattr(namespace, dest, None)
        if value is None:
            continue
        if action.option_strings:
            if isinstance(value, bool):
                if value:
                    arguments.append(action.option_strings[0])
            else:
                for option_string in action.option_strings:
                    arguments.append(option_string)
                arguments.append(str(value))
        else:
            arguments.append(str(value))
    return ' '.join(arguments)



if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--launch_mode", type=str, default="main",
                            choices=["main"], help="Launch mode")
        args, extra_args = parser.parse_known_args()

        if args.launch_mode == "main":
            main(extra_args)
    except Exception as e:
        logger = get_logger()
        logger.exception(e)
        exit(1)
    exit(0)
