import os
import time
from typing import List

try:
    import pytorch_lightning as pl
    from pytorch_lightning.utilities import rank_zero_info
except ImportError:
    raise ImportError(
        "Please install pytorch-lightning for using data modules: "
        "`pip install pytorch-lightning`"
    )

import torch.utils.data as data
import torchvision
from torchvision.datasets import CIFAR10, ImageFolder, VOCDetection

import torch

import bcos.settings as settings

from .categories import CIFAR10_CATEGORIES, IMAGENET_CATEGORIES, PASCALVOC_CATEGORIES
from .sampler import RASampler
from .transforms import RandomCutmix, RandomMixup, SplitAndGrid

__all__ = ["ImageNetDataModule", "CIFAR10DataModule", "ClassificationDataModule", "PASCALVOCDataModule"]


class ClassificationDataModule(pl.LightningDataModule):
    """Base class for data modules for classification tasks."""

    NUM_CLASSES: int = None
    """Number of classes in the dataset."""
    NUM_TRAIN_EXAMPLES: int = None
    """Number of training examples in the dataset. Need not be defined."""
    NUM_EVAL_EXAMPLES: int = None
    """Number of evaluation examples in the dataset. Need not be defined."""
    CATEGORIES: List[str] = None
    """List of categories in the dataset. Need not be defined."""

    # ===================================== [ Registry stuff ] ======================================
    __data_module_registry = {}
    """Registry of data modules."""

    def __init_subclass__(cls, **kwargs):
        # check that the class attributes are defined
        super().__init_subclass__(**kwargs)
        assert cls.NUM_CLASSES is not None
        # rest don't need to be defined

        # get name and remove DataModule suffix
        name = cls.__name__
        # check if name matches XXXDataModule
        if not name.endswith("DataModule"):
            raise ValueError(
                f"Data module class name '{name}' does not end with 'DataModule'"
            )
        name = name[: -len("DataModule")]
        # check if name is already registered
        if name in cls.__data_module_registry:
            raise ValueError(f"Data module {name} already registered")
        # register the class in the registry
        cls.__data_module_registry[name] = cls

    @classmethod
    def registry(cls):
        """Returns the registry of data modules."""
        return cls.__data_module_registry

    # ===================================== [ Normal stuff ] ======================================
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]

        self.train_dataset = None
        self.eval_dataset = None

        mixup_alpha = config.get("mixup_alpha", 0.0)
        cutmix_alpha = config.get("cutmix_alpha", 0.0)
        p_gridified = config.get("p_gridified", 0.0)
        self.train_collate_fn = self.get_train_collate_fn(
            mixup_alpha, cutmix_alpha, p_gridified
        )

    def train_dataloader(self):
        train_sampler = self.get_train_sampler()
        shuffle = None if train_sampler is not None else True
        return data.DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=self.num_workers,
            collate_fn=self.train_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.eval_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.eval_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

    @classmethod
    def get_train_collate_fn(
        cls,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        p_gridified: float = 0.0,
    ):
        assert not (p_gridified and mixup_alpha), "For now, do not use both."

        collate_fn = None
        if p_gridified:
            gridify = SplitAndGrid(p_gridified, num_classes=cls.NUM_CLASSES)

            def collate_fn(batch):
                return gridify(*data.default_collate(batch))

            rank_zero_info(f"Gridify active for training with {p_gridified=}")

        mixup_transforms = []
        if mixup_alpha > 0.0:
            mixup_transforms.append(
                RandomMixup(cls.NUM_CLASSES, p=1.0, alpha=mixup_alpha)
            )
            rank_zero_info(f"Mixup active for training with {mixup_alpha=}")
        if cutmix_alpha > 0.0:
            mixup_transforms.append(
                RandomCutmix(cls.NUM_CLASSES, p=1.0, alpha=cutmix_alpha)
            )
            rank_zero_info(f"Cutmix active for training with {cutmix_alpha=}")
        if mixup_transforms:
            mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

            def collate_fn(batch):  # noqa: F811
                return mixupcutmix(*data.default_collate(batch))

        return collate_fn

    def get_train_sampler(self):
        train_sampler = None

        # see https://github.com/Lightning-AI/lightning/blob/612d43e5bf38ba73b4f372d64594c2f9a32e6d6a/src/pytorch_lightning/trainer/connectors/data_connector.py#L336
        # and https://github.com/Lightning-AI/lightning/blob/612d43e5bf38ba73b4f372d64594c2f9a32e6d6a/src/lightning_lite/utilities/seed.py#L54
        seed = int(os.getenv("PL_GLOBAL_SEED", 0))
        ra_reps = self.config.get("ra_repetitions", None)
        if ra_reps is not None:
            rank_zero_info(f"Activating RASampler with {ra_reps=}")
            train_sampler = RASampler(
                self.train_dataset,
                shuffle=True,
                seed=seed,
                repetitions=ra_reps,
            )

        return train_sampler


class ImageNetDataModule(ClassificationDataModule):
    # from https://image-net.org/download.php
    NUM_CLASSES: int = 1000

    NUM_TRAIN_EXAMPLES: int = 34_745
    NUM_EVAL_EXAMPLES: int = 50_000

    CATEGORIES: List[str] = IMAGENET_CATEGORIES

    def __init__(self, config):
        super().__init__(config)
        self.prepare_data_per_node = self.config.get("cache_dataset", None) == "shm"

    def prepare_data(self) -> None:
        cache_dataset = self.config.get("cache_dataset", None)
        if cache_dataset != "shm":
            return

        # print because we also want global non-zero rank's
        start = time.perf_counter()
        print("Caching dataset into SHM!...")
        from .caching import cache_tar_files_to_shm

        cache_tar_files_to_shm()
        end = time.perf_counter()
        print(f"Caching successful! Time taken {end - start:.2f}s")

    def setup(self, stage: str) -> None:
        # this way changes to the settings are reflected at function call time
        SHMTMPDIR = settings.SHMTMPDIR
        IMAGENET_PATH = settings.IMAGENET_PATH
        if stage == "fit":
            cache_dataset = self.config.get("cache_dataset", None)
            rank_zero_info("Setting up ImageNet train dataset...")
            start = time.perf_counter()
            train_root = os.path.join(
                SHMTMPDIR if cache_dataset == "shm" else IMAGENET_PATH,
                "train",
            )
            self.train_dataset = ImageFolder(
                root=train_root,
                transform=self.config["train_transform"],
            )
            assert len(self.train_dataset) == self.NUM_TRAIN_EXAMPLES
            rank_zero_info(f"Done! Took time {time.perf_counter() - start:.2f}s")

            if cache_dataset == "onthefly":
                rank_zero_info("Trying to setup Bagua's cached dataset!")
                from .caching import CachedImageFolder

                self.train_dataset = CachedImageFolder(self.train_dataset)
                rank_zero_info("Successfully setup cached dataset!")

        start = time.perf_counter()
        rank_zero_info("Setting up ImageNet val dataset...")
        self.eval_dataset = ImageFolder(
            root=os.path.join(IMAGENET_PATH, "val"),
            transform=self.config["test_transform"],
        )
        assert len(self.eval_dataset) == self.NUM_EVAL_EXAMPLES
        rank_zero_info(f"Done! Took time {time.perf_counter() - start:.2f}s")


class CIFAR10DataModule(ClassificationDataModule):
    # from https://www.cs.toronto.edu/~kriz/cifar.html
    NUM_CLASSES: int = 10

    NUM_TRAIN_EXAMPLES: int = 50_000
    NUM_EVAL_EXAMPLES: int = 578#10_000 TODO: modify this 

    CATEGORIES: List[str] = CIFAR10_CATEGORIES

    def setup(self, stage: str) -> None:
        DATA_ROOT = settings.DATA_ROOT
        ADVERSARIAL = os.environ.get('ADVERSARIAL')
        if stage == "fit":
            self.train_dataset = CIFAR10(
                root=DATA_ROOT,
                train=True,
                transform=self.config["train_transform"],
                download=True,
            )
            assert len(self.train_dataset) == self.NUM_TRrAIN_EXAMPLES
        if ADVERSARIAL: 
            self.eval_dataset = ImageFolder(
                root='bcos/data/cifar10_adversarial_epsilon_03/adversarial',
                transform=self.config["test_transform"],
            )
        else: 
            self.eval_dataset = CIFAR10(
                root=DATA_ROOT,
                train=False,
                transform=self.config["test_transform"],
                download=True,
            )
        #assert len(self.eval_dataset) == self.NUM_EVAL_EXAMPLES TODO: put this back in 


class VOCDetectionClassification(VOCDetection):
    CLASS_TO_IDX = {name: idx for idx, name in enumerate(PASCALVOC_CATEGORIES)}
    
    def __getitem__(self, idx):
        image, target = super().__getitem__(idx)
        
        objects = target['annotation']['object']
        
        # Ensure objects are always a list
        if not isinstance(objects, list):
            objects = [objects]
        
        # Find the object with the largest bounding box
        largest_area = 0
        largest_label = None
        for obj in objects:
            bbox = obj['bndbox']
            xmin, ymin = int(bbox['xmin']), int(bbox['ymin'])
            xmax, ymax = int(bbox['xmax']), int(bbox['ymax'])
            
            # Calculate area of the bounding box
            area = (xmax - xmin) * (ymax - ymin)
            if area > largest_area:
                largest_area = area
                largest_label = obj['name']
        
        return image, self.CLASS_TO_IDX[largest_label]


class PascalVOCDatasetROI(VOCDetection):
    CLASS_TO_IDX = {name: idx for idx, name in enumerate(PASCALVOC_CATEGORIES)}
    
    def __getitem__(self, idx):
        image, target = super().__getitem__(idx)

        # Parse VOC annotations into the format required
        objects = target["annotation"]["object"]
        if not isinstance(objects, list):
            objects = [objects]

        boxes = []
        labels = []
        for obj in objects:
            bbox = obj["bndbox"]
            boxes.append([
                float(bbox["xmin"]),
                float(bbox["ymin"]),
                float(bbox["xmax"]),
                float(bbox["ymax"]),
            ])
            labels.append(self.CLASS_TO_IDX[obj["name"]])

        return image, {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
            }


class PASCALVOCDataModule(ClassificationDataModule):
    # from https://www.cs.toronto.edu/~kriz/cifar.html
    NUM_CLASSES: int = 20

    NUM_TRAIN_EXAMPLES: int = 5_717
    NUM_EVAL_EXAMPLES: int = 5_823

    CATEGORIES: List[str] = PASCALVOC_CATEGORIES

    def setup(self, stage: str) -> None:
        PASCALVOC_PATH = os.environ.get("PASCALVOC_PATH")
        self.ROI = os.environ.get("ROI", "false").lower() == 'true'
        VOCDATASET = PascalVOCDatasetROI if self.ROI else VOCDetectionClassification
        if stage == "fit":
            self.train_dataset = VOCDATASET(
                root=PASCALVOC_PATH,
                year="2012",
                image_set="train",
                download=True,
                transforms=self.config["train_transform"]
            )
            assert len(self.train_dataset) == self.NUM_TRAIN_EXAMPLES

        self.eval_dataset = VOCDATASET(
            root=PASCALVOC_PATH,
            year="2012",
            image_set="val",
            download=True,
            transforms=self.config["test_transform"]
        )
        assert len(self.eval_dataset) == self.NUM_EVAL_EXAMPLES
