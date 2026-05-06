import os
import io
import torch
import decord
import pathlib
import asyncio
import threading
import numpy as np
from PIL import Image
from datasets import Image as DImage
from typing import Tuple, Union, List, Generator, Optional

from innerT2V.dataset.utils import download_aidata_content


VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', 'bmp')
VALID_VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.webm', '.mkv')


def load_image(path):
    if isinstance(path, dict):
        if 'path' in path and path['path'].startswith('oss://'):
            img = download_aidata_content(path)
            if not isinstance(img, Image.Image):
                img = Image.open(io.BytesIO(img["bytes"]))
        else:
            img = DImage().decode_example(path)
    else:
        img = Image.open(path)
    img = np.array(img, dtype=np.float32)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=2)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def get_frame_indices(num_frames, video_length, sample='uniform'):
    if sample == 'uniform':
        acc_samples = min(num_frames, video_length)
        intervals = np.linspace(start=0, stop=video_length, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        if len(frame_indices) < num_frames:
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    else:
        raise NotImplementedError
    return frame_indices


class VideoStreamReader:

    def __init__(self, video_path: dict | str | pathlib.Path, height: int = -1, width: int = -1):
        if isinstance(video_path, dict) and 'path' in video_path:
            import io
            from datasets import Video
            video_reader = decord.VideoReader(uri=io.BytesIO(Video().decode_example(video_path)['bytes']))
            video_num_frames = len(video_reader)
        else:
            video_path = pathlib.Path(video_path) if not isinstance(video_path, pathlib.Path) else video_path
            if video_path.is_dir():
                frame_files = sorted([
                    file.name for file in video_path.iterdir() if file.is_file() and file.suffix.lower() in VALID_IMAGE_EXTENSIONS])
                video_num_frames = len(frame_files)
                frames_extractor = lambda indices: torch.stack(
                    [torch.from_numpy(
                        load_image(video_path.joinpath(frame_file))) for frame_file in map(frame_files.__getitem__, indices)])
                video_reader = None
            else:
                video_reader = decord.VideoReader(uri=video_path.as_posix(), num_threads=1)
                video_num_frames = len(video_reader)

        if video_reader is not None:
            def safe_get_batch(reader, indices):
                ret = reader.get_batch(indices)
                reader.seek(0)
                return ret
            frames_extractor = lambda indices: safe_get_batch(video_reader, indices)

        self._video_reader = video_reader

        self.video_path = video_path
        self.video_num_frames = video_num_frames
        self.frames_extractor = frames_extractor
        self.height = height
        self.width = width

    def close(self):
        try:
            if self._video_reader is not None:
                del self._video_reader
        except:
            pass

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def __len__(self) -> int:
        return self.video_num_frames

    def __getitem__(self, indices: Union[List[int], int]) -> torch.Tensor:
        if isinstance(indices, int):
            indices = [indices]
        decord.bridge.set_bridge("torch")
        frames = self.frames_extractor(indices)
        frames = frames.float().div(255.).clip(0, 1)
        frames = frames.permute(0, 3, 1, 2).contiguous()
        if self.height > 0 and self.width > 0:
            frames = torch.nn.functional.interpolate(frames, size=(self.height, self.width), mode='bicubic', antialias=True)
            frames = frames.clip(0, 1)
        return frames

    def get_fps(self) -> int:
        try:
            return self._video_reader.get_avg_fps()
        except:
            return None

    def resolution(self) -> Tuple[int, int]:
        tmp_frame = self[0]
        return tmp_frame.shape[-2], tmp_frame.shape[-1]

    def extract_all(self) -> torch.Tensor:
        return self[list(range(self.video_num_frames))]

    def extract_sampling(self, num_frames: int = 100, mode: str = 'uniform') -> torch.Tensor:
        return self[get_frame_indices(num_frames, self.video_num_frames, mode)]

    def extract_clips(
        self,
        num_tile_frames: int = 121,
        num_overlap_frames: int = 65,
        frame_begin: int = 0,
        frame_end: int = -1,
    ) -> Generator[torch.Tensor, None, None]:

        return ClipExtractor(self, num_tile_frames, num_overlap_frames, frame_begin, frame_end)


class ClipExtractor:

    def __init__(
        self,
        reader: VideoStreamReader,
        num_tile_frames: int = 121,
        num_overlap_frames: int = 65,
        frame_begin: int = 0,
        frame_end: int = -1,
    ):
        self.reader = reader
        self.num_tile_frames = num_tile_frames
        self.num_overlap_frames = num_overlap_frames

        self.num_stride_frames = num_tile_frames - num_overlap_frames

        if frame_begin < 0:
            frame_begin = 0
        self.frame_begin = frame_begin
        if frame_end == -1 or frame_end > len(self.reader):
            frame_end = len(self.reader)
        self.frame_end = frame_end
        self.num_frames = frame_end - frame_begin

        num_clips = (self.num_frames - self.num_tile_frames) // self.num_stride_frames + 1
        if (num_clips - 1) * self.num_stride_frames + self.num_tile_frames < self.num_frames:
            num_clips += 1
        self.num_clips = max(1, num_clips)

    def __getitem__(self, i: int) -> torch.Tensor:
        idx_begin = self.frame_begin + i * self.num_stride_frames
        idx_end = min(idx_begin + self.num_tile_frames, self.frame_end)
        indices = list(range(idx_begin, idx_end))
        return self.reader[indices]

    def __iter__(self):
        for i in range(self.num_clips):
            yield self[i]

    def __len__(self):
        return self.num_clips


class VideoStreamWriter:

    def __init__(self, video_path: str, height: int, width: int, fps: int = 10, crf: int = 18,
                 ref_video_path: Optional[str] = None, is_disabled_rank: bool = False):
        self.writing_proc = None
        if not is_disabled_rank:
            ffmpeg_path = os.getenv('FFMPEG_PATH', None)
            assert ffmpeg_path, "FFMPEG_PATH is not set"
            subenv = os.environ.copy()
            subenv['LD_LIBRARY_PATH'] = f"{os.path.dirname(ffmpeg_path)}:{subenv['LD_LIBRARY_PATH']}"

            ac_args = ''
            if ref_video_path is not None and os.path.splitext(ref_video_path)[1].lower() in VALID_VIDEO_EXTENSIONS:
                ac_args = f'-i {ref_video_path} -map 1:a? '

            cmd = f'{ffmpeg_path} -f rawvideo -pix_fmt rgb24 -s {width}x{height} -r {fps} -i - {ac_args} -map 0:v ' \
                f'-c:v libs265 -pix_fmt yuv420p -color_range 1 -colorspace bt709 -color_trc bt709 -color_primaries bt709 ' \
                f'-s265-params limit-refs=0:b-adapt=3:bframes=15:crf={crf}:psnr=1:ssim=1:cbqpoffs=-1:crqpoffs=-1:aq-mode=1:aq-strength=1.1:tune-ssim=1:dynamic-crf=1:dcrf-usecrf=1:info=0:keyint=250 ' \
                '-threads:v 4 -qmin 0 -qmax 45 -preset veryslow -c:a copy -movflags +faststart ' \
                f'-map_metadata -1 -loglevel warning -tune ssim -tag:v hvc1 {video_path} -y'

            def _init_coroutine_loop(coroutine_loop):
                coroutine_loop.run_forever()
                coroutine_loop.close()
            self.coroutine_loop = asyncio.new_event_loop()
            self.coroutine_thread = threading.Thread(target=_init_coroutine_loop, args=(self.coroutine_loop, ))
            self.coroutine_thread.start()
            self.writing_proc = asyncio.run_coroutine_threadsafe(
                asyncio.create_subprocess_shell(cmd, stdin=asyncio.subprocess.PIPE, env=subenv),
                self.coroutine_loop
            ).result()
        self.height = height
        self.width = width

    def close(self):
        if self.writing_proc is None: return

        async def _aclose(proc):
            proc.stdin.write_eof()
            await proc.stdin.wait_closed()
            await proc.communicate()

        asyncio.run_coroutine_threadsafe(
            _aclose(self.writing_proc),
            self.coroutine_loop
        ).result()
        self.coroutine_loop.call_soon_threadsafe(self.coroutine_loop.stop)
        self.coroutine_thread.join()

        self.writing_proc = None

    async def write_frame(self, frame):
        if self.writing_proc is None: return
        if isinstance(frame, np.ndarray):
            frame = frame.astype(np.uint8)
        elif isinstance(frame, Image.Image):
            frame = np.array(frame)
        elif isinstance(frame, torch.Tensor):
            frame = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        frame = np.array(Image.fromarray(frame).resize((self.width, self.height), Image.Resampling.LANCZOS))
        self.writing_proc.stdin.write(frame.tobytes())
        await self.writing_proc.stdin.drain()

    def write_clip(self, clip):
        if self.writing_proc is None: return
        for frame in clip:
            asyncio.run_coroutine_threadsafe(
                self.write_frame(frame),
                self.coroutine_loop
            ).result()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __del__(self):
        self.close()
