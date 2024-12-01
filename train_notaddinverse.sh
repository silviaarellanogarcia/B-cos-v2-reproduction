ADDINVERSE="false" DATA_ROOT="../cifar10/" B_PARAMETER="1.25" MAXOUT_PARAMETER="2" python train.py --dataset CIFAR10 --base_network norm_ablations_final --experiment_name resnet_20_lnu-linear > results/add-inverse/resnet_20_lnu-linear-train.txt
ADDINVERSE="false" DATA_ROOT="../cifar10/" B_PARAMETER="1.25" MAXOUT_PARAMETER="2" python evaluate.py --dataset CIFAR10 --experiment_name resnet_20_lnu-linear --base_network norm_ablations_final --reload last > results/add-inverse/resnet_20_lnu-linear-eval.txt
ADDINVERSE="false" DATA_ROOT="../cifar10/" B_PARAMETER="1.25" MAXOUT_PARAMETER="2" python -m interpretability.analyses.localisation --reload best_any --analysis_config 500_3x3 --explainer_name Ours --smooth 15 --batch_size 64 --save_path "./experiments/CIFAR10/norm_ablations_final/resnet_20_lnu-linear/" > results/add-inverse/resnet_20_lnu-linear-loc.txt