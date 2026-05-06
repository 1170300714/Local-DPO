import time
from typing import Optional
from collections import defaultdict

from .logger import get_logger

logger = get_logger()


class Timer:

    _default_process_name = 'default'

    def __init__(self, average_period: int = 1):
        self.average_period = average_period
        self.clear()

    def clear(self):
        self._durations = defaultdict(list)
        self._start_times = []

    def tic(self, name: Optional[str] = None):
        self._start_times.append((name or self._default_process_name, time.time()))

    def toc(self):
        name, start_time = self._start_times.pop(-1)
        self._durations[name].append(time.time() - start_time)
        if len(self._durations[name]) % self.average_period == 0:
            logger.info(f"[Timer] {name} - {sum(self._durations[name]) / len(self._durations[name]):.4f} s")
            self._durations.pop(name)

    def __call__(self, name: Optional[str] = None):
        setattr(self, '_input_name', name)
        return self

    def __enter__(self):
        name = getattr(self, '_input_name', None)
        delattr(self, '_input_name')
        self.tic(name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.toc()
