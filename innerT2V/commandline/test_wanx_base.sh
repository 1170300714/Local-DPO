RESOURCEDIR=XXX

job_name=job
worker_count=1

EXP_ROOT=XXX
FRAME=49
HEIGHT=480
WIDTH=720

N_VIDEOS_PER_PROMPT=1


PROMPT_ARGS="\
    --prompts_file=XXX \
"


data_args="\
    ${PROMPT_ARGS} \
    --output_dir=XXX \
    --fps=24 \
    --height=${HEIGHT} \
    --width=${WIDTH} \
    --sample_frames=${FRAME} \
"



model_args="\
    --base_modules_dir=XXX \
    --tuned_modules_dir=XXX \
    --num_videos_per_prompt=${N_VIDEOS_PER_PROMPT} \
    --seed 42 \
    --add_pos_prompt \
    --add_neg_prompt \
"



args="--config_file=./nebula_configs/accelerate_configs.yaml --num_processes ${worker_count} \
    innerT2V/test_wanx21.py ${model_args} ${data_args}"
