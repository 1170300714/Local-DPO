import json
import numpy as np
from tqdm import tqdm
from typing import List


class EfficientDataMetaHandler(object):

    def __init__(self, meta_files: List[str]):

        self.meta_files = meta_files

        assert all([meta_file.endswith('.jsonl') for meta_file in meta_files]), "Only support jsonl file."

        file_reading_infos = {}
        file_offset = 0
        for meta_file in tqdm(meta_files):
            bytes_offset = []
            with open(meta_file, 'rb') as f:
                for line in tqdm(f):
                    bytes_offset.append(len(line))
            bytes_offset = [0] + np.cumsum(bytes_offset).tolist()

            file_reading_infos[meta_file] = {
                'nlines': len(bytes_offset) - 1,
                'file_offset': file_offset,
                'line_offsets': bytes_offset,
            }
            file_offset += len(bytes_offset) - 1
        self.file_reading_infos = file_reading_infos

        self.n_total_lines = sum([file_reading_info['nlines'] for file_reading_info in self.file_reading_infos.values()])

        self._masking_indices = set()
        self._indices_mapping = {}

    def set_masking_indices(self, indices: List[int]):
        self._masking_indices = set(indices)
        self._indices_mapping = {}

        if len(self._masking_indices) == 0:
            return

        new_i = 0
        for i in range(self.n_total_lines):
            if i in self._masking_indices: continue
            self._indices_mapping[new_i] = i
            new_i += 1

    def get_meta_index(self, index: int) -> int:
        return self._indices_mapping.get(index, index)

    def __len__(self) -> int:
        return self.n_total_lines - len(self._masking_indices)

    def __getitem__(self, index: int) -> dict:
        index = self._indices_mapping.get(index, index)
        for meta_file, info in self.file_reading_infos.items():
            if index >= info['file_offset'] and index < info['file_offset'] + info['nlines']:
                index -= info['file_offset']
                line_offsets = info['line_offsets']
                with open(meta_file, 'rb') as f:
                    f.seek(line_offsets[index])
                    line = f.read(line_offsets[index + 1] - line_offsets[index])
                return json.loads(line)
        return None

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
