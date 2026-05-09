worker_count=$1
OUTPUT_DIR=$2
FRAME=$3
HEIGHT=$4
WIDTH=$5
DATA_INFO=$6
BASE_DIR=$7
N_VIDEOS_PER_PROMPT=1
yita_min=0.85
yita_max=0.95
data_args="\
    --data_info ${DATA_INFO} \
    --output_dir=${OUTPUT_DIR} \
    --fps=8 \
    --height=${HEIGHT} \
    --width=${WIDTH} \
    --sample_frames=${FRAME} \
"

model_args="\
    --base_modules_dir=${BASE_DIR} \
    --num_videos_per_prompt=${N_VIDEOS_PER_PROMPT} \
    --add_pos_prompt \
    --add_neg_prompt \
    --yita_min ${yita_min} \
    --yita_max ${yita_max} \
"

args=" --num_processes ${worker_count} \
    innerT2V/generate_corrupted_videos.py ${model_args} ${data_args}"

echo "Args is: ${args}"