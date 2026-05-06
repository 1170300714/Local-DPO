RESOURCEDIR=/mnt/workspace/qidong/concept_learning
# RESOURCEDIR=/data/oss_bucket_0/Users/haomin/mount


worker_count=$1

EXP_ROOT=$2

FRAME=$3
HEIGHT=$4
WIDTH=$5

N_VIDEOS_PER_PROMPT=$6

ENHANCE="long" # Qwen-V0.4, short, gpt4o, gpt4o_wo_scenecut_v2, deepseek_v1

PROMPT_ARGS="\
    --prompts_file=$7 \
"

BASE_DIR=$8
TUNED_DIR=$9

data_args="\
    ${PROMPT_ARGS} \
    --output_dir=${EXP_ROOT}/evaluation/ \
    --fps=8 \
    --height=${HEIGHT} \
    --width=${WIDTH} \
    --sample_frames=${FRAME} \
"

model_args="\
    --base_modules_dir=$BASE_DIR \
    --tuned_modules_dir=$TUNED_DIR \
    --prompt_expander ${ENHANCE} \
    --num_videos_per_prompt=${N_VIDEOS_PER_PROMPT} \
    --seed 42 \
    --add_pos_prompt \
    --add_neg_prompt \
"

args=" --num_processes ${worker_count} \
    innerT2V/test.py ${model_args} ${data_args}"

echo "Args is: ${args}"