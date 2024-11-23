import argparse
from pathlib import Path
from PIL import Image
import os
import matplotlib.pyplot as plt
from bcos.data.categories import IMAGENET_CATEGORIES

import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = lambda x: x  # noqa: E731

import os
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np
import cv2

from bcos.data.datamodules import ClassificationDataModule
from bcos.experiments.utils import Experiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model.", add_help=add_help
    )

    # specify save dir and experiment config
    parser.add_argument(
        "--hubconf",
        help="Test model from local hubconf file.",
    )

    parser.add_argument(
        "--base_directory",
        default="./experiments",
        help="The base directory.",
    )
    parser.add_argument(
        "--dataset",
        choices=["ImageNet", "CIFAR10", "PASCALVOC"],
        default="ImageNet",
        help="The dataset.",
    )
    parser.add_argument(
        "--base_network", help="The model config or base network to use."
    )
    parser.add_argument("--experiment_name", help="The name of the experiment to run.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--reload", help="What ckpt to load. ['last', 'best', 'epoch_<N>', 'best_any']"
    )
    group.add_argument(
        "--weights",
        metavar="PATH",
        type=Path,
        help="Specific weight state dict to load.",
    )

    parser.add_argument(
        "--ema",
        default=False,
        action="store_true",
        help="Load the EMA stored version if it exists. Not applicable for reload='best_any'.",
    )

    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size to use. Default is 1"
    )
    parser.add_argument(
        "--no-cuda",
        default=False,
        action="store_true",
        help="Force into not using cuda.",
    )

    return parser


def run_evaluation(args):
    global device
    if args.no_cuda:
        device = torch.device("cpu")

    if device == torch.device("cuda"):
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # get model, config, and data
    model, config = load_model_and_config(args)
    test_loader = get_test_loader(args.dataset, config)

    # do evaluation
    ADVERSARIAL_PLOT = os.environ.get("ADVERSARIAL_PLOT", "false").lower() == 'true'
    if ADVERSARIAL_PLOT:
        evaluate_explanation(model, test_loader)
    else:
        ROI = os.environ.get("ROI", "false").lower() == 'true'
        eval_funct = evaluate_mAP if ROI else evaluate
        eval_funct(model, test_loader)


def evaluate_explanation(model, data_loader):
    explanation_save_dir = "bcos/data/imagenet_mini_explanation_05"
    os.makedirs(explanation_save_dir, exist_ok=True)

    model.eval()

    for batch_idx, (image, target) in enumerate(tqdm(data_loader)):       
        image = image.to(device, non_blocking=True)
        image.requires_grad = True
        
        # CODE TO DISPLAY BOTH ORIGINAL AND MODIFIED CIFAR10 
        '''
        original_image = image.squeeze(0).detach().cpu()  # Convert to HWC format
        if original_image.shape[0] > 3:  # More than 3 channels
            original_image = original_image[:3, :, :].permute(1, 2, 0).numpy()  # Use only the first 3 channels
        
        output = model.explain(image)
        explanation = output['explanation']

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(explanation, cmap='viridis')  # Optional: Use a colormap for the explanation
        axes[1].set_title("Explanation")
        axes[1].axis('off')
        
        image_save_path = os.path.join(explanation_save_dir, f"explanation_{batch_idx}.png")
        plt.savefig(image_save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        '''
        
        # CODE TO DISPLAY ONLY MODFIED IMAGE IMAGENET 

        output = model.explain(image)
        explanation = output['explanation']
        predicted_index = output["prediction"]
        predicted_label = IMAGENET_CATEGORIES[predicted_index]
        # Plot the explanation
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(explanation, cmap="viridis")  # Display the explanation
        ax.axis("off")
        
        # Add the label as text at the bottom
        plt.text(
            0.5, -0.1, predicted_label, fontsize=12, ha="center", transform=ax.transAxes
        )
        
        # Save the explanation
        image_save_path = os.path.join(explanation_save_dir, f"explanation_{batch_idx}.png")
        plt.savefig(image_save_path, bbox_inches="tight", pad_inches=0.5)
        plt.close(fig)

       

def evaluate_mAP(model, data_loader):
    from PIL import ImageDraw, Image
    import matplotlib.pyplot as plt
    model.eval()
    map_metric = MeanAveragePrecision()
    total_samples = 0
    for image, target in tqdm(data_loader):
        image = image.to(device, non_blocking=True)

        output = model.explain(image)
        final_mask = output['explanation'][..., 3]
        # gray_scale = output['explanation'][..., :3].mean(axis=-1)
        
        # final_mask = (gray_scale + alpha_channel) / 2.0
        rois_pred = {
            "boxes": [],
            "scores": [],
            "labels": [],
        }
        img = np.transpose(image[0][:3].detach().cpu().numpy(), (1, 2, 0))
        img = (img * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        for thresh in np.arange(0.95, 1.0, 0.05):
            thresh_mask = final_mask > thresh
            contours, _ = cv2.findContours(thresh_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x,y,w,h = cv2.boundingRect(contour)
                rois_pred["boxes"].append([x, y, x + w, y + h])
                rois_pred["scores"].append(thresh)
                rois_pred["labels"].append(target['labels'][0][0].item())
                draw.rectangle((x,y,x+w,y+h), outline="green", width=10)
        plt.imshow(np.array(img))
        plt.savefig("example_plot.png")
        plt.imshow(final_mask)
        plt.savefig("example_plot2.png")
        
        rois_pred["boxes"] = torch.tensor(rois_pred["boxes"], dtype=torch.float32)
        rois_pred["scores"] = torch.tensor(rois_pred["scores"], dtype=torch.float32)
        rois_pred["labels"] = torch.tensor(rois_pred["labels"], dtype=torch.int64)
                
        total_samples += image.shape[0]
        target = {
            'labels': target['labels'][0],
            'boxes': target['boxes'][0],
        }
        map_metric.update([rois_pred], [target])
        if total_samples > 250:
            break
        
    map_result = map_metric.compute()
    print(
        f"Out of a total of {total_samples}, mAP result:"
    )
    print()
    print("--------------------------------------------")
    print(map_result)
    print("--------------------------------------------")
    print()


def evaluate(model, data_loader):
    # https://github.com/pytorch/vision/blob/657c0767c5ca5564c8b437ac44263994c8e0/references/classification/train.py#L61
    model.eval()

    total_samples = 0
    total_correct_top1 = 0
    total_correct_top5 = 0
    with torch.inference_mode():
        for image, target in tqdm(data_loader):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(image)

            total_samples += image.shape[0]
            correct_top1, correct_top5 = check_correct(output, target, topk=(1, 5))
            total_correct_top1 += correct_top1.item()
            total_correct_top5 += correct_top5.item()

    acc1 = total_correct_top1 / total_samples
    acc5 = total_correct_top5 / total_samples
    print(
        f"Out of a total of {total_samples}, got {total_correct_top1=} and {total_correct_top5=}"
    )
    print()
    print("--------------------------------------------")
    print(f"Acc@1 {acc1:.3%} Acc@5 {acc5:.3%}")
    print("--------------------------------------------")
    print()


def check_correct(output, target, topk=(1,)):
    with torch.inference_mode():
        maxk = max(topk)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum()
            res.append(correct_k)
        return res


def load_model_and_config(args):
    # a bit messy because of trying to directly use hubconf
    if args.hubconf is not None:
        import hubconf

        model = getattr(hubconf, args.hubconf)(pretrained=True)
        config = model.config
    else:
        experiment = Experiment(
            base_directory=args.base_directory,
            path_or_dataset=args.dataset,
            base_network=args.base_network,
            experiment_name=args.experiment_name,
        )
        config = experiment.config

        if args.reload is not None:
            model = experiment.load_trained_model(
                reload=args.reload, verbose=True, ema=args.ema
            )
        elif args.weights is not None:
            model: torch.nn.Module = experiment.get_model()
            state_dict = torch.load(args.weights, map_location="cpu")
            try:
                model.load_state_dict(state_dict)
            except RuntimeError as e:
                print(
                    "Error loading state dict. Please note that --weights only supports "
                    "loading model state dict and not from training checkpoints."
                )
                raise e
        else:
            raise RuntimeError(
                "One of --reload, --weights or --hubconf must be provided!"
            )

    model = model.to(device)

    return model, config


def get_test_loader(dataset, config):
    registry = ClassificationDataModule.registry()
    if dataset in registry:
        datamodule = registry[dataset](config["data"])
    else:
        available_datasets = list(registry.keys())
        raise ValueError(
            f"Unknown dataset: '{dataset}'. Available datasets are: {available_datasets}"
        )

    # get data and set batchsize
    datamodule.batch_size = args.batch_size
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    return test_loader


if __name__ == "__main__":
    args = get_parser().parse_args()
    run_evaluation(args)
