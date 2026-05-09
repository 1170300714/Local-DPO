worker_count=$1
OUTPUT_DIR=$2
FRAME=${3:-49}
HEIGHT=${4:-480}
WIDTH=${5:-720}
N_VIDEOS_PER_PROMPT=$6


PROMPT_ARGS="\
    --prompts_file=$7 \
"
BASE_DIR=$8
TUNED_DIR=$9

data_args="\
    ${PROMPT_ARGS} \
    --output_dir=${OUTPUT_DIR} \
    --fps=24 \
    --height=${HEIGHT} \
    --width=${WIDTH} \
    --sample_frames=${FRAME} \
"



model_args="\
    --base_modules_dir=${BASE_DIR} \
    --tuned_modules_dir=$TUNED_DIR \
    --num_videos_per_prompt=${N_VIDEOS_PER_PROMPT} \
    --seed 42 \
    --add_pos_prompt \
    --add_neg_prompt \
"



args="--num_processes ${worker_count} \
    innerT2V/test.py ${model_args} ${data_args}"
