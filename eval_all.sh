#!/bin/bash

# List of ResNet models to evaluate
# RESNET_MODELS=("resnet50_long" "resnet152_long" "convnext_tiny" "convnext_base" "resnet101" "vgg11_bnu" "densenet161")
RESNET_MODELS=("convnext_tiny")
# Loop through each model
for model in "${RESNET_MODELS[@]}"; do
    echo "Evaluating model: $model"
    # Run the evaluation and redirect output to a corresponding text file
    IMAGENET_PATH="../imagenet_validation/" python evaluate.py --dataset ImageNet --hubconf $model > "results/accuracy/${model}.txt"
done

echo "All evaluations completed."
