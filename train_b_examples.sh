#!/bin/bash

# List of ResNet models to evaluate
BCOS=("1" "1.25" "1.50" "1.75" "2.00" "2.25" "2.50")

export GPU_NUMBER=0
# Loop through each model
for value in "${BCOS[@]}"; do
    echo "Training with B: $value"
    # Run the evaluation and redirect output to a corresponding text file
    DATA_ROOT="../cifar10/" B_PARAMETER="$value" MAXOUT_PARAMETER="2" python train.py --dataset CIFAR10 --base_network norm_ablations_final --experiment_name resnet_20 > results/train-b/${value}-train.txt
    DATA_ROOT="../cifar10/" B_PARAMETER="$value" MAXOUT_PARAMETER="2" python evaluate.py --dataset CIFAR10 --experiment_name resnet_20 --base_network norm_ablations_final --reload last > results/train-b/${value}-eval.txt
    DATA_ROOT="../cifar10/" B_PARAMETER="$value" MAXOUT_PARAMETER="2" python -m interpretability.analyses.localisation --reload best_any --analysis_config 500_3x3 --explainer_name Ours --smooth 15 --batch_size 64 --save_path "./experiments/CIFAR10/norm_ablations_final/resnet_20/" > results/train-b/${value}-loc.txt
    mv experiments/CIFAR10/norm_ablations_final/resnet_20/last.ckpt checkpoints/resnet20/${value}.ckpt
    rm experiments/CIFAR10/norm_ablations_final/resnet_20/*
done

echo "All training completed."
