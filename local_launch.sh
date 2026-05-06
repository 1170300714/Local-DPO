task_name=$1
# arguments from the caller script will be directly passed to sourced script, so we shift to bypass the 1st arg (task name)
shift;
source ./innerT2V/commandline/${task_name}_base.sh

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    NCCL_DEBUG=INFO \
    TORCH_DISTRIBUTED_DEBUG=INFO \
    XFORMERS_FORCE_DISABLE_TRITON=1 \
    accelerate launch ${args}
