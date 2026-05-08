import os
import io
import sys

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../')))

import json
import math
import random
import jsonlines
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
from pipeline.pipeline_cogvideox_improved_dense_dpo_mask import CogVideoXImprovedPipeline
from utils import summarize_model_info

import logging
from utils.logger import get_logger, add_handler, set_default_formatter
from utils.random_mask_gen import create_random_shape_with_random_motion_multiple_connected_components

import torchvision.transforms as TT
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
import numpy as np
from datasets import Video

from dataset.utils import download_aidata_content


import decord
decord.bridge.set_bridge("torch")


VALID_TUNED_MODULES = {
    'text_encoder': T5EncoderModel,
    'transformer': CogVideoXImprovedTransformer3DModel,
    'vae': AutoencoderKLCogVideoX,
}

SYSTEM_POSITIVE_PROMPTS = "highly detailed, perfect without deformations, ultra HD, "
SYSTEM_NEGATIVE_PROMPTS = "blurring, dirty, messy, low quality, cartoon, drawing, anime"


class T2VDataset():
    def __init__(
        self,
        data_info: str, 
        frame_sampling_mode: str = "interval",
        resize_mode: str = "center",
        frame_num: int = 49,
        height: int = 768,
        width: int = 1360,
        min_step: int = 1,
        max_step: int = 2,
        add_pos_prompt: bool = True,
        add_neg_prompt: bool = True,
        add_fps_prompt: bool = False,
    ):
        self.data_info = data_info
        self.frame_sampling_mode = frame_sampling_mode
        self.resize_mode = resize_mode
        self.frame_num = frame_num
        self.height = height
        self.width = width
        self.min_step = min_step
        self.max_step = max_step
        self.frame_num = frame_num
        self.transforms = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.float().div(255.0)), 
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True), 
            ]
        )
        self.add_pos_prompt = add_pos_prompt
        self.add_neg_prompt = add_neg_prompt
        self.add_fps_prompt = add_fps_prompt

    def __len__(self):
        return len(self.data_info)

    def __iter__(self):
        for idx, d in enumerate(self.data_info):
            video_path = d['video_path']
            frames, motion, fps = self._load_video(video_path, self.frame_num)
            if self.resize_mode != 'none':
                frames = self._resize_for_rectangle_crop(frames, (self.height, self.width))
            else:
                frames = resize(frames, (self.height, self.width))
            frames = self.transforms(frames)
            prompt = d['qwen25vl7b_caption_2']
            file_name = d['vid']

            prompt, neg_prompt = self.prepare_prompt(prompt, self.add_pos_prompt, self.add_neg_prompt, self.add_fps_prompt)

        
            package_data = {
                'video': frames,
                'fps': fps,
                'metadata': {
                    'raw_metadata': d,
                    'index': idx,
                    'num_frames': self.frame_num,
                    'height': self.height,
                    'width': self.width,
                },
                'prompt': prompt,
                'neg_prompt': neg_prompt,
                'file_name': file_name,
            }
            yield package_data

    def _load_video(self, video_path, n_target_frames):
        video_obj = video_path
        if isinstance(video_path, dict):
            if 'path' in video_path and video_path['path'].startswith('oss://'):
                target_video = download_aidata_content(video_path)
            else:
                target_video = Video().decode_example(video_path)
            video_bytes = target_video['bytes']
            video_obj = io.BytesIO(video_bytes)
        elif isinstance(video_path, str):
            with open(video_path, 'rb') as f:
                video_bytes = f.read()
            video_obj = io.BytesIO(video_bytes)
        video_reader = decord.VideoReader(uri=video_obj, num_threads=8)
        fps = video_reader.get_avg_fps()
        video_num_frames = len(video_reader)
        frame_indices, sampling_interval = self._get_indices(video_len=video_num_frames, sample_num=n_target_frames)

        frames = video_reader.get_batch(frame_indices)[:n_target_frames].float().permute(0, 3, 1, 2).contiguous()

     
        if frames.shape[0] < n_target_frames:
            frames = torch.cat([frames, frames[-1:].repeat(n_target_frames - frames.shape[0], 1, 1, 1)], dim=0)

        motion = None
        fps = (fps + sampling_interval - 1) // sampling_interval
        return frames, motion, fps

    def _get_indices(self, video_len, sample_num):
        if sample_num >= video_len - 2:
            return list(range(video_len)), 1

        if not getattr(self, '_info_frame_sampling', False):
            setattr(self, '_info_frame_sampling', True)
            print(f'[Config] Frame sampling mode: {self.frame_sampling_mode}')

        if self.frame_sampling_mode == 'uniform':
 
            step_size = video_len // sample_num
            indices = list(range(0, video_len, step_size))
        elif self.frame_sampling_mode == 'continuous':
     
            center_frame = video_len // 2
            indices = list(range(center_frame - sample_num // 2, center_frame + sample_num // 2 + 1))
            indices = list(filter(lambda x: x >= 0 and x < video_len, indices))
            step_size = 1
        elif self.frame_sampling_mode == 'prev_2s':
            indices = list(range(0, 49))
            indices = list(filter(lambda x: x >= 0 and x < video_len, indices))
            step_size = 1
        elif self.frame_sampling_mode == 'interval':
            while True:
                step_size = np.random.randint(self.min_step, self.max_step)
                max_start_idx = video_len - step_size * (sample_num - 1) - 1
                max_start_idx = max(max_start_idx, 0)
                start_idx = np.random.randint(0, max_start_idx)
                end_idx = start_idx + step_size * (sample_num - 1)
                if end_idx > video_len - 1:
                    continue
                indices = range(start_idx, end_idx + 1, step_size)
                break
        else:
            raise NotImplementedError
        return indices[:sample_num], step_size

    def _resize_for_rectangle_crop(self, arr, image_size):
        reshape_mode = self.resize_mode
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC
            )
        h, w = arr.shape[2], arr.shape[3]
 

        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        if reshape_mode == 'random' or reshape_mode == 'none':
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == 'center':
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        return arr

    def prepare_prompt(self, prompt, add_pos_prompt, add_neg_prompt, add_fps_prompt):
        prompt = prompt.strip()
        if add_pos_prompt:
            prompt = f'{SYSTEM_POSITIVE_PROMPTS}{prompt}'
        if add_fps_prompt:
            fps = self.data_info.get('fps', 24)
            prompt = f'FPS-{fps}. {prompt}'

        neg_prompt = None
        if add_neg_prompt:
            neg_prompt = SYSTEM_NEGATIVE_PROMPTS

        return prompt, neg_prompt


def mask_2_pt(all_masks):
    all_masks_pt = []
    counter = 0
    mask_len = len(all_masks[0])
    for i in range(1, len(all_masks)):
        assert len(all_masks[i]) == mask_len, f'{i} mask len: {len(all_masks[i])}, 0 mask len: {mask_len}'
    for j in range(mask_len):
        mask = np.array(all_masks[0][j])
        for i in range(1, len(all_masks)):
            mask = mask + np.array(all_masks[i][j])
        mask[mask >= 1] = 1

        mask = torch.from_numpy(mask)
        all_masks_pt.append(mask)
        counter += 1
    all_masks_pt = torch.stack(all_masks_pt, dim=0)
    return all_masks_pt
    

def sample_with_interval(min_val, max_val, interval):


    choices = np.arange(min_val, max_val + 1e-9, interval)
    

    return np.random.choice(choices)


def main(cml_args=None, only_get_args: bool = False):
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_info", type=str, required=True, help="data info for dense-dpo video preparation")
    parser.add_argument("--output_dir", type=str,required=True, help="Output directory for the generated video")
    parser.add_argument("--log_path", type=str, default=None, help="Log file path. If not set, use stdout")

    parser.add_argument("--base_modules_dir", type=str, default="THUDM/CogVideoX1.5-5B")
    parser.add_argument("--tuned_modules_dir", type=str, default=None)
    parser.add_argument("--lora_scale", type=float, default=1.0, help="Lora scale")

    parser.add_argument('--frame_sampling_mode', type=str, default='prev_2s')
    parser.add_argument('--resize_mode', type=str, default='center')
    parser.add_argument("--fps", type=int, default=24, help="Frames per second for the output video")
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=720)
    parser.add_argument('--sample_frames', type=int, default=49)
    parser.add_argument('--min_step', type=int, default=1)
    parser.add_argument('--max_step', type=int, default=2)
    parser.add_argument('--num_videos_per_prompt', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)


    parser.add_argument('--add_fps_prompt', '--add_fps_prefix', action='store_true', default=False)
    parser.add_argument('--add_pos_prompt', action='store_true', default=False)
    parser.add_argument('--add_neg_prompt', action='store_true', default=False)

    parser.add_argument('--enable_parallel', action='store_true')
    parser.add_argument('--yita_min', type=float, default=0.75)
    parser.add_argument('--yita_max', type=float, default=0.85)

    args, _ = parser.parse_known_args(cml_args)

    if args.log_path is not None:
        handler = logging.FileHandler(args.log_path, mode='a')
        add_handler(handler)
    set_default_formatter()
    logger = get_logger()

    data = []
    with open(args.data_info, 'r', encoding='utf8') as f:
        for item in jsonlines.Reader(f):
            data.append(item)


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
            data = [data[i::n_chunks] for i in range(n_chunks)][get_data_parallel_rank()]
    else:
        n_chunks = accelerator.num_processes
        data = [data[i::n_chunks] for i in range(n_chunks)][accelerator.process_index]
    logger.info(f"Assign {len(data)} data items to rank {accelerator.process_index}.")

    pipe.enable_model_cpu_offload(device=accelerator.device)


    dataset = T2VDataset(data, args.frame_sampling_mode, args.resize_mode, 
                        args.sample_frames, args.height, args.width, args.min_step, args.max_step, 
                        args.add_pos_prompt, args.add_neg_prompt, args.add_fps_prompt)

    os.makedirs(args.output_dir, exist_ok=True)

    output_mask = os.path.join(args.output_dir, 'masks')
    output_original_video = os.path.join(args.output_dir, 'original_videos')
    output_edited_video = os.path.join(args.output_dir, 'edited_videos')

    os.makedirs(output_mask, exist_ok=True)
    os.makedirs(output_original_video, exist_ok=True)
    os.makedirs(output_edited_video, exist_ok=True)

    mask_length, mask_height, mask_width = int(args.sample_frames // 4) + 1, args.height, args.width

    max_cc_num = 5 
    min_cc_num = 1  

    for _, d in tqdm(enumerate(dataset)):
        frames = d['video'].unsqueeze(0)
        frames = frames.to(accelerator.device)
        cur_cc_num = random.randint(min_cc_num, max_cc_num)
        fix_area = random.uniform(0, 1)
        if fix_area < 0.7:
            fix_area = 1
        else:
            fix_area = 0
        cc_ratio = math.sqrt(cur_cc_num)
        all_candidate_masks = []
        print(f'cur_cc_num: {cur_cc_num}, fix_area: {fix_area}')
        for _ in range(cur_cc_num):
            candidate_masks = create_random_shape_with_random_motion_multiple_connected_components(mask_length, zoomin=0.9, zoomout=1.1, rotmin=1, rotmax=10, cc_ratio=cc_ratio, fix_area=fix_area, imageHeight=mask_height, imageWidth=mask_width)
            all_candidate_masks.append(candidate_masks)
        mask_torch = mask_2_pt(all_candidate_masks)
        # temp = input('kkpsa')
        mask = mask_torch.to(accelerator.device)
        prompt = d['prompt']
        neg_prompt = d['neg_prompt']
        file_name = d['file_name']
        # temp = input(f'filename: {file_name}')
        sample_frames = d['metadata']['height'], d['metadata']['width'], d['metadata']['num_frames']

        if len(file_name) > 1000:
            file_name = file_name[:1000]
            logger.warning(f"File name is too long, truncated to 500 characters.")
        is_absent = False
        for i in range(args.num_videos_per_prompt):
            output_file = os.path.join(output_mask, f"{file_name}-{i}.pt")
            if not os.path.exists(output_file):
                is_absent = True
        is_absent = is_absent and not os.path.exists(os.path.join(output_mask, f"{file_name}.pt"))
        if not is_absent:
            logger.info(f"Skip {prompt}")
            continue

        logger.info(f"Prompt is: {prompt}")
        logger.info(f"Negative prompt is: {neg_prompt}")



        sampled_yita = sample_with_interval(args.yita_min, args.yita_max, 0.05)

        videos1_output, frames_output = pipe(
            frames=frames,
            prompt=prompt,
            negative_prompt=neg_prompt,
            num_videos_per_prompt=args.num_videos_per_prompt,
            height=args.height,
            width=args.width,
            num_frames=sample_frames,
            use_dynamic_cfg=True,
            guidance_scale=6.0,
            generator=torch.Generator().manual_seed(args.seed),
            yita=sampled_yita,
            mask=mask,
        )
        videos1 = videos1_output.frames
        frames_output = frames_output.frames
        if (args.enable_parallel and is_dp_last_group()) or not args.enable_parallel:

            for i, video in enumerate(videos1):
                output_file = os.path.join(output_edited_video, f"{file_name}-{i}-yita-{sampled_yita}-cc-{cur_cc_num}-fix-{fix_area}.mp4")
                original_video = os.path.join(output_original_video, f'{file_name}-{i}.mp4')
                mask_outpath = os.path.join(output_mask, f"{file_name}-{i}.pt")
                if args.num_videos_per_prompt == 1:
                    output_file = os.path.join(output_edited_video, f"{file_name}-yita-{sampled_yita}-cc-{cur_cc_num}-fix-{fix_area}.mp4")
                    original_video = os.path.join(output_original_video, f'{file_name}.mp4')
                    mask_outpath = os.path.join(output_mask, f"{file_name}.pt")

                tmp_file1 = f'/tmp/{os.path.basename(output_file)}'
                tmp_file2 = f'/tmp/{os.path.basename(original_video)}'
                tmp_file3 = f'/tmp/{os.path.basename(mask_outpath)}'
                export_to_video(video, tmp_file1, fps=args.fps)
                export_to_video(frames_output[0], tmp_file2, fps=args.fps)
                torch.save(mask_torch, tmp_file3)
                shutil.move(tmp_file1, output_file)
                shutil.move(tmp_file2, original_video)
                shutil.move(tmp_file3, mask_outpath)





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

        main(extra_args)
    except Exception as e:
        logger = get_logger()
        logger.exception(e)
        exit(1)
    exit(0)
