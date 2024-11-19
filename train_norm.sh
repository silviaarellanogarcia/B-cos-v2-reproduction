#!/bin/bash

# List of ResNet models to evaluate
EXTRA=("" "-linear")
NORM=("an" "bn" "gn" "in" "ln" "pn" "anu" "bnu" "gnu" "inu" "lnu" "pnu")

export GPU_NUMBER=0
# Loop through each model
for e_val in "${EXTRA[@]}"; do
    for value in "${NORM[@]}"; do
        echo "Training with B: $value"
        # Run the evaluation and redirect output to a corresponding text file
        DATA_ROOT="../cifar10/" B_PARAMETER="1.25" MAXOUT_PARAMETER="2" python train.py --dataset CIFAR10 --base_network norm_ablations_final --experiment_name resnet_20_${value}${e_val} > results/train-norm/${value}${e_val}-train.txt
        DATA_ROOT="../cifar10/" B_PARAMETER="1.25" MAXOUT_PARAMETER="2" python evaluate.py --dataset CIFAR10 --experiment_name resnet_20_${value}${e_val} --base_network norm_ablations_final --reload last > results/train-norm/${value}${e_val}-eval.txt
        DATA_ROOT="../cifar10/" B_PARAMETER="1.25" MAXOUT_PARAMETER="2" python -m interpretability.analyses.localisation --reload best_any --analysis_config 500_3x3 --explainer_name Ours --smooth 15 --batch_size 64 --save_path "./experiments/CIFAR10/norm_ablations_final/resnet_20_${value}${e_val}/" > results/train-norm/${value}${e_val}-loc.txt
    done
done

echo "All training completed."
