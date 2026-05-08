import os
import io
import torch
import decord
import numpy as np
from tqdm import tqdm
from functools import reduce
from typing import List, Optional, Union, Dict
from collections import OrderedDict

import torchvision.transforms as TT
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
import torch.nn.functional as F
from datasets import Video

from .efficient_data_meta_handler import EfficientDataMetaHandler
from utils.misc import Timer

decord.bridge.set_bridge("torch")

from utils.logger import get_logger

logger = get_logger()

DEFAULT_CAPTION_KEYS = 'gen_caption'

class T2VDataset(Dataset):
    def __init__(
        self,
        dataset_meta: Union[List[Dict], EfficientDataMetaHandler],
        resize_mode: str = "center",
        height_buckets: int = 768,
        width_buckets: int = 1360,
        frame_buckets: List[int] = None,
        random_flip: Optional[float] = None,
        min_step: int = 1,
        max_step: int = 8,
        candidate_caption_keys: List[str] = [DEFAULT_CAPTION_KEYS],
        candidate_caption_weights: List[float] = [1.0],
        unconditional_probs: float = 0.0,
        add_fps_prefix: bool = True,
        use_tags_as_prompt_prob: float = 0.0,
        use_gen_prompt_prob: float = 0.0,
        use_humanlong_prompt_prob: float = 0.0,
        shift_image: bool = False,
        frame_sampling_mode='interval',
        data_cache_dir: Optional[str] = None,
        buckets = None,
        calculate_vmafmotion: bool = False,
        camera_motion_aug_prob: float = 0.0,
    ) -> None:
        super().__init__()

        self.dataset_meta = dataset_meta

        assert len(height_buckets) == 1, 'Height bucket len: {len(height_buckets)}, should be 1'
        assert len(width_buckets) == 1, 'Width bucket len: {len(width_buckets)}, should be 1'
        self.height = height_buckets[0]
        self.width = width_buckets[0]
        self.frame_buckets = frame_buckets

        self.candidate_caption_keys = candidate_caption_keys or [DEFAULT_CAPTION_KEYS]
        candidate_caption_weights = candidate_caption_weights or [1.0]
        self.candidate_caption_weights = np.array(candidate_caption_weights) / sum(candidate_caption_weights)

        self.resize_mode = resize_mode
        self.frame_sampling_mode = frame_sampling_mode
        self.min_step = min_step
        self.max_step = max_step
        assert self.max_step >= self.min_step

        self.buckets = self._prepare_bucket(buckets)
        logger.info(f"[Config] Dataset buckets: {self.buckets}")

        self.transforms = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.float().div(255.0)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        self._timer = Timer(10)

    def __len__(self) -> int:
        return len(self.dataset_meta)

    def _prepare_bucket(self, buckets=None):
        if buckets is None:
            t_config = {
                1: (1.0, None),
            }
            for f in self.frame_buckets:
                t_config[f] = (1.0, None)
            buckets = {
                f'{self.height}x{self.width}': t_config
            }

        def accumulate_prod(xs):
            return reduce(lambda x, y: int(x) * int(y), xs, 1)
        buckets = OrderedDict(
            sorted(buckets.items(), key=lambda x: accumulate_prod(x[0].split('x')), reverse=True))
        for hw_id in buckets:
            buckets[hw_id] = OrderedDict(sorted(buckets[hw_id].items(), key=lambda x: x[0], reverse=True))
        return buckets

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
    
    def _get_meta_info(self, index, seed=None):

        rng = np.random.default_rng(seed)

        data_info = self.dataset_meta[index]
        T = min(data_info.get('pos_num_frames', 0),data_info.get('neg_num_frames', 0))
        if T is None:
            duration = data_info.get('duration', 0) or 0
            fps = data_info.get('fps', 30) or 30
            T = int(duration * fps)

        H = data_info.get('height', 1080)
        W = data_info.get('width', 1920)

        for hw_id, t_config in self.buckets.items():
            th, tw = map(int, hw_id.split('x'))
            if H * W < th * tw: continue

            if (H > W and th < tw) or (H < W and th > tw):
                th, tw = tw, th

            if T == 1 or T == 0:
                if 1 in t_config:
                    if rng.random() < t_config[1][0]:
                        t = 1
                        if rng.random() < self.camera_motion_aug_prob:
                            t = rng.choice(list(t_config.keys()), size=1)[0]
                        return {
                            "num_frames": t,
                            "height": th,
                            "width": tw,
                            "batch_scale": t_config[t][1]
                        }

            for t, (prob, batch_scale) in t_config.items():
                if T >= t and t != 1 and rng.random() < prob:
                    return {
                        "num_frames": t,
                        "height": th,
                        "width": tw,
                        "batch_scale": batch_scale
                    }

        th = int(min((H // 16) * 16, 768))
        tw = int(min((W // 16) * 16, 1360))
        t = int(min(max(T, 1), 121))
        batch_scale = 1
        logger.warning(f'Cannot find proper bucket, force to {t}-{th}-{tw}-{batch_scale}, info is\n{data_info}')
        return {
            "num_frames": t,
            "height": th,
            "width": tw,
            "batch_scale": batch_scale,
        }

    def _get_indices(self, video_len, sample_num):
        if sample_num >= video_len - 2:
            return list(range(video_len)), 1

        if not getattr(self, '_info_frame_sampling', False):
            setattr(self, '_info_frame_sampling', True)
            logger.info(f'[Config] Frame sampling mode: {self.frame_sampling_mode}')

        if self.frame_sampling_mode == 'uniform':
            step_size = video_len // sample_num
            indices = list(range(0, video_len, step_size))
        elif self.frame_sampling_mode == 'continuous':
            center_frame = video_len // 2
            indices = list(range(center_frame - sample_num // 2, center_frame + sample_num // 2 + 1))
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

    def _load_video(self, video_path, n_target_frames):
        video_obj = video_path
        if isinstance(video_path, dict):
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

    def _load_content(self, path, n_target_frames):
        return self._load_video(path, n_target_frames)

    def get_prompt(self, info, nframes=-1):
        default_weight = 0
        if 'gen_caption' in self.candidate_caption_keys:
            i = self.candidate_caption_keys.index('gen_caption')
            default_weight = self.candidate_caption_weights[i]

        valid_keys = ['gen_caption']
        valid_weights = [default_weight]

        for key, w in zip(self.candidate_caption_keys, self.candidate_caption_weights):
            if key == 'gen_caption': continue
            if info.get(key, None) is None:
                valid_weights[0] += w
            else:
                valid_keys.append(key)
                valid_weights.append(w)

        if len(valid_keys) == 1:
            key = valid_keys[0]
        else:
            valid_weights = np.array(valid_weights) / sum(valid_weights)
            key = np.random.choice(valid_keys, p=valid_weights)
        prompt = info.get(key, None)

        return prompt, key

    def _resize_mask(self, mask, new_height, new_width, mode):
        mask = mask.unsqueeze(1)
        mask = F.interpolate(
            mask, 
            size=(new_height, new_width), 
            mode=mode, 
            align_corners=False if mode in ('linear', 'bilinear', 'bicubic', 'trilinear') else None
        )
        mask[mask > 0.5] = 1
        return mask


    def __getitem__(self, index):
        index, n_target_frames, height, width = list(map(int, index.split('-')))

        data_info = self.dataset_meta[index]
        pos_target_path = data_info['pos_video_path']
        neg_target_path = data_info['neg_video_path']
        mask = data_info['mask']
        if 'yita' in data_info:
            yita = data_info['yita']
        else:
            yita = None

        mask = torch.load(mask)
        h_factor, w_factor = 8, 8

        try:
            with self._timer('load_content'):
                pos_frames, pos_motion, pos_fps = self._load_content(pos_target_path, n_target_frames=n_target_frames)
                neg_frames, neg_motion, neg_fps = self._load_content(neg_target_path, n_target_frames=n_target_frames)
            if self.resize_mode != 'none':
                pos_frames = self._resize_for_rectangle_crop(pos_frames, (height, width))
                neg_frames = self._resize_for_rectangle_crop(neg_frames, (height, width))
            else:
                pos_frames = resize(pos_frames, (height, width))
                neg_frames = resize(neg_frames, (height, width))
            pos_frames = self.transforms(pos_frames)
            neg_frames = self.transforms(neg_frames)
            pos_nframes = pos_frames.shape[0]
            pos_height = pos_frames.shape[2]
            pos_width = pos_frames.shape[3]
            neg_nframes = neg_frames.shape[0]
            neg_height = neg_frames.shape[2]
            neg_width = neg_frames.shape[3]
            if (
                pos_nframes != neg_nframes or
                pos_height != neg_height or
                pos_width != neg_width or
                pos_motion != neg_motion
            ):
                raise ValueError(
                    f"Mismatch between positive and negative samples for index {index}: "
                    f"pos (fps={pos_fps}, frames={pos_nframes}, height={pos_height}, width={pos_width}), "
                    f"neg (fps={neg_fps}, frames={neg_nframes}, height={neg_height}, width={neg_width})."
                )
            fps = pos_fps
            nframes = pos_nframes
            height = pos_height
            width = pos_width

            mask = self._resize_mask(mask, height // h_factor, width // w_factor, 'nearest')

        except Exception as e:
            logger.error(f"Error loading video {pos_target_path} or {neg_target_path} with Exception {e}")
            fps = -1
            nframes = -1
            height = -1
            width = -1
        prompt, prompt_key = self.get_prompt(data_info, nframes)
        
        meta_index = index
        if hasattr(self.dataset_meta, 'get_meta_index'):
            meta_index = self.dataset_meta.get_meta_index(index)
        return {
                'pos_video': pos_frames,
                'neg_video': neg_frames,
                "mask": mask,
                "yita": yita,
                'fps': fps,
                'vid': data_info.get('vid', pos_target_path),
                'metadata': {
                    'raw_metadata': data_info,
                    'index': meta_index,
                    'num_frames': nframes,
                    'height': height,
                    'width': width,
                },
                'prompt': prompt,
                'prompt_key': prompt_key,
            }
