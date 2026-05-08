from typing import Dict, Any

class SACStorage:

    def __init__(self):
        self._enabled: bool = False
        self._storage: Dict[str, Any] = {}

    def __getitem__(self, key):
        if not self._enabled: return None
        return self._storage[key]

    def __contains__(self, key):
        if not self._enabled: return False
        return key in self._storage

    def __setitem__(self, key, value):
        if not self._enabled: return
        self._storage[key] = value

    def enable(self, enabled: bool = True):
        self._enabled = enabled

    def pop(self, key):
        if not self._enabled: return None
        return self._storage.pop(key)


sac_storage = SACStorage()
