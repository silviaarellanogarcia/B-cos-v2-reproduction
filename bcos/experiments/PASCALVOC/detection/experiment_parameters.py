import math
from functools import partial

import torch.nn as nn

from bcos.data.presets import (
    PASCALVOCClassificationPresetEval,
    PASCALVOCClassificationPresetTrain,
)
from bcos.experiments.utils import (
    configs_cli,
    create_configs_with_different_seeds,
    update_config,
)
from bcos.modules import norms
from bcos.modules.losses import BinaryCrossEntropyLoss
from bcos.optim import LRSchedulerFactory, OptimizerFactory

import os

B_PARAMETER = float(os.environ.get("B_PARAMETER", "2"))
MAXOUT_PARAMETER = int(os.environ.get("MAXOUT_PARAMETER", "2"))

__all__ = ["CONFIGS"]

NUM_CLASSES = 20

# This config is based on
# https://github.com/moboehle/B-cos/blob/main/experiments/CIFAR10/bcos/experiment_parameters.py
DEFAULT_NUM_EPOCHS = 100
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 16
DEFAULT_CROP_SIZE = 224

DEFAULTS = dict(
    data=dict(
        train_transform=PASCALVOCClassificationPresetTrain(is_bcos=True, crop_size=DEFAULT_CROP_SIZE),
        test_transform=PASCALVOCClassificationPresetEval(is_bcos=True, crop_size=DEFAULT_CROP_SIZE),
        batch_size=DEFAULT_BATCH_SIZE,
        num_workers=4,
        num_classes=NUM_CLASSES,
    ),
    model=dict(
        is_bcos=True,
        # "name": None,
        args=dict(
            num_classes=NUM_CLASSES,
        ),
        bcos_args=dict(
            b=B_PARAMETER,
        ),
        max_out_args=dict(
            maxout=MAXOUT_PARAMETER,
        )
    ),
    criterion=BinaryCrossEntropyLoss(),
    test_criterion=BinaryCrossEntropyLoss(),
    optimizer=OptimizerFactory(name="Adam", lr=DEFAULT_LR),
    lr_scheduler=LRSchedulerFactory(
        name="cosineannealinglr", epochs=DEFAULT_NUM_EPOCHS
    ),
    trainer=dict(
        max_epochs=DEFAULT_NUM_EPOCHS,
    ),
)


# helper
def update_default(new_config):
    return update_config(DEFAULTS, new_config)


RESNET_DEPTHS = [20, ]
NORMS_MAP = {
    # centered
    "an": norms.centered_norms.AllNorm2d,
    "bn": norms.centered_norms.BatchNorm2d,
    "gn": partial(norms.centered_norms.DetachableGroupNorm2d, 2),  # num_groups
    "in": norms.centered_norms.DetachableGNInstanceNorm2d,
    "ln": norms.centered_norms.DetachableGNLayerNorm2d,
    "pn": norms.centered_norms.DetachablePositionNorm2d,
    # uncentered
    "anu": norms.uncentered_norms.AllNormUncentered2d,
    "bnu": norms.uncentered_norms.BatchNormUncentered2d,
    "gnu": partial(norms.uncentered_norms.GroupNormUncentered2d, 2),
    "inu": norms.uncentered_norms.GNInstanceNormUncentered2d,
    "lnu": norms.uncentered_norms.GNLayerNormUncentered2d,
    "pnu": norms.uncentered_norms.PositionNormUncentered2d,
}

resnets = {
    f"resnet_{d}": update_default(
        dict(
            model=dict(
                name=f"resnet{d}",
                args=dict(
                    norm_layer=nn.Identity,
                    logit_bias=-math.log(NUM_CLASSES - 1),
                ),
                bcos_args=dict(
                    max_out=2,
                ),
            ),
        )
    )
    for d in RESNET_DEPTHS
}

resnets_norms = {
    f"{name}_{norm_name}{affine_or_not_suffix}": update_config(
        old_c,
        dict(
            model=dict(
                args=dict(
                    norm_layer=affine_or_not_wrapper(norm_layer),
                ),
            ),
        ),
    )
    for name, old_c in resnets.items()
    for norm_name, norm_layer in NORMS_MAP.items()
    for affine_or_not_suffix, affine_or_not_wrapper in [
        ("", lambda x: x),
        ("-linear", norms.NoBias),
        ("-unaffine", norms.Unaffine),
    ]
}

resnets_nomaxout = {
    f"{name}-nomaxout": update_config(
        old_c,
        dict(
            model=dict(
                bcos_args=dict(
                    max_out=1,
                )
            )
        ),
    )
    for config_group in [resnets, resnets_norms]
    for name, old_c in config_group.items()
}


CONFIGS = dict()
CONFIGS.update(resnets)
CONFIGS.update(resnets_norms)
CONFIGS.update(resnets_nomaxout)
CONFIGS.update(create_configs_with_different_seeds(CONFIGS, seeds=[420, 1337]))


if __name__ == "__main__":
    configs_cli(CONFIGS)
