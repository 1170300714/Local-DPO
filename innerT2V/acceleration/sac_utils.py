from typing import Dict, Any

r'''[NOTE] Selective Activation Checkpointing (SAC) for ring attention
Under activation checkpointing context, ctx.save_for_backward will be skip during forward;
during backward, the forward call will be conducted again to trigger ctx.save_for_backward.
SAC can be used to prevent the recomputation for certain operations, which reduce the time consumption
at a cost of extra memory. Thus it is suitable for some computation-bounded operations such as attention.
However, up to torch-2.7, official torch's SAC mechanism cannot be used with custom autograd.Function
since it depends on __torch_dispatch__ which cannot recognize custom autograd functions.

In order to achieve SAC on custom attention, we do a hack here by manually saving and restoring the output through a global storage.
So during the second forward call triggered by backward, the output and softmax_lse will be retrieved from the global storage directly
without recomputation. A problem here is that there is no way to distinguish between the first and second forward call 
in official torch.autograd.Function.
We now use `sac_id` (specified by the outer module) as a unique identifier for each SAC call, which requires some ugly monkey-patching.

[TODO] A better solution is to implement some tricky context managers to manage the whole activation checkpointing.
'''

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
