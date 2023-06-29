
model_root="/home/jovyan/runs/metrabs-exp/"



declare -a arr=(
    "partial_ms_s1/128in_j8"
#    "/path/to/checkpoint/dir2"
#    "/path/to/checkpoint/dir3"
    # Add more directories as needed
)

# Run the command for each directory
for dir in "${arr[@]}"; do
    ./main.py --predict --dataset=h36m --checkpoint-dir="${model_root}/${dir}" --backbone=mobilenetV3Smalls1 --init=scratch --mobilenet-alpha=1.0 --model-class=Metro --final-transposed-conv=1 --stride-test=32 --upper-bbox --upper-bbox-ratio 0.5 0.5 --proc-side=128 --output-upper-joints
    python -m eval_scripts.eval_h36m --pred-path="${model_root}/${dir}"/predictions_h36m.npz --root-last
    ./main.py --predict --dataset=h36m --checkpoint-dir="${model_root}/${dir}" --backbone=mobilenetV3Smalls1 --init=scratch --mobilenet-alpha=1.0 --model-class=Metro --final-transposed-conv=1 --stride-test=32 --proc-side=128 --output-upper-joints
    python -m eval_scripts.eval_h36m --pred-path="${model_root}/${dir}"/predictions_h36m.npz --root-last    
done