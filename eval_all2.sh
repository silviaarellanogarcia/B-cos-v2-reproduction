#!/bin/bash

# List of ResNet models to evaluate
# RESNET_MODELS=("resnet50_long" "resnet152_long" "convnext_tiny" "convnext_base" "resnet101" "vgg11_bnu" "densenet161")
RESNET_MODELS=("resnet_18" "resnet_34" "resnet_50" "resnet_101" "resnet_151" "resnet_50_32x4d" "densenet_121" "densenet_161" "densenet_169" "densenet_201" "vgg_11_bnu")
export GPU_NUMBER=1
# Loop through each model
for model in "${RESNET_MODELS[@]}"; do
    echo "Evaluating model: $model"
    # Run the evaluation and redirect output to a corresponding text file
    IMAGENET_PATH="../imagenet_validation/" python -m interpretability.analyses.localisation --reload best_any --analysis_config 500_3x3 --explainer_name Ours --smooth 15 --batch_size 64 --save_path "./bcos/experiments/ImageNet/bcos_final/${model}/" > "results/localisation/${model}.txt"
done

RESNET_MODELS_LONG=("resnet_50" "resnet_152" "densenet_121" "convnext_tiny_bnu" "convnext_base_bnu" "convnext_tiny_pn" "convnext_base_pn")

for model in "${RESNET_MODELS_LONG[@]}"; do
    echo "Evaluating model: $model"
    # Run the evaluation and redirect output to a corresponding text file
    IMAGENET_PATH="../imagenet_validation/" python -m interpretability.analyses.localisation --reload best_any --analysis_config 500_3x3 --explainer_name Ours --smooth 15 --batch_size 64 --save_path "./bcos/experiments/ImageNet/bcos_final_long/${model}/" > "results/localisation/${model}_long.txt"
done

VIT_MODELS=("bcos_simple_vit_ti_patch16_224" "bcos_simple_vit_s_patch16_224" "bcos_simple_vit_b_patch16_224" "bcos_simple_vit_l_patch16_224" "bcos_vitc_s_patch1_14" "bcos_vitc_ti_patch1_14" "bcos_vitc_b_patch1_14" "bcos_vitc_l_patch1_14")

for model in "${VIT_MODELS[@]}"; do
    echo "Evaluating model: $model"
    # Run the evaluation and redirect output to a corresponding text file
    IMAGENET_PATH="../imagenet_validation/" python -m interpretability.analyses.localisation --reload best_any --analysis_config 500_3x3 --explainer_name Ours --smooth 15 --batch_size 64 --save_path "./bcos/experiments/ImageNet/vit_final/${model}/" > "results/localisation/${model}.txt"
done

echo "All evaluations completed."
