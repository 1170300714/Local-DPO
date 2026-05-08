import torch
from typing import Dict, Any, List


class VideoCollateFunction:

    def __init__(self, weight_dtype: torch.dtype = torch.float32) -> None:
        self.weight_dtype = weight_dtype

    def __call__(self, data: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pos_is_valids =  [x["pos_video"] is not None for x in data]
        neg_is_valids =  [x["neg_video"] is not None for x in data]
        if pos_is_valids != neg_is_valids:
            return None
        is_valids = pos_is_valids 
        pos_videos, neg_videos, fps_list, metas, indices, masks = [], [], [], [], [], []
        for i, x in enumerate(data):
            if not is_valids[i] : continue
            pos_videos.append(x["pos_video"])
            neg_videos.append(x["neg_video"])
            masks.append(x['mask'])
            fps_list.append(x.get("fps", 30))
            metas.append(x.get("metadata", {}).get("raw_metadata", {}))
            indices.append(x.get("metadata", {}).get("index", -1))

        if len(pos_videos) == 0 or len(neg_videos) == 0: return None
        pos_videos = torch.stack(pos_videos).to(dtype=self.weight_dtype, non_blocking=True)
        neg_videos = torch.stack(neg_videos).to(dtype=self.weight_dtype, non_blocking=True)
        masks = torch.stack(masks).to(dtype=self.weight_dtype, non_blocking=True)
        return {
            "pos_videos": pos_videos,
            "neg_videos": neg_videos,
            "fps": fps_list,
            "masks": masks,
            "metas": metas,
            "indices": indices,
        }

class T2VCollateFunction(VideoCollateFunction):

    def __call__(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        ret = super().__call__(data)
        pos_is_valids =  [x["pos_video"] is not None for x in data]
        neg_is_valids =  [x["neg_video"] is not None for x in data]
        if pos_is_valids != neg_is_valids:
            return None
        is_valids = pos_is_valids 
        prompts = []
        yitas = []
        for i, x in enumerate(data):
            if not is_valids[i]: continue
            prompts.append(x["prompt"])
            yitas.append(float(x['yita']))
        yitas = torch.tensor(yitas)
        ret.update({
            "prompts": prompts,
            'yitas': yitas
        })
        return ret
