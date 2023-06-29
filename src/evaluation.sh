
model_root="/home/jovyan/runs/metrabs-exp/"



declare -a arr=(
    "partial_ms_s1/112in_j8"
#    "/path/to/checkpoint/dir2"
#    "/path/to/checkpoint/dir3"
    # Add more directories as needed
)

# Run the command for each directory
for dir in "${arr[@]}"; do
    ./main.py --predict --dataset=h36m --checkpoint-dir="${model_root}/${dir}" --backbone=mobilenetV3Small --init=scratch --mobilenet-alpha=1.0 --model-class=Metro --final-transposed-conv=1 --stride-test=32 --upper-bbox --upper-bbox-ratio 0.5 0.5 --proc-side=160 --output-upper-joints
done