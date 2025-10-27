import os

import pandas as pd  # noqa: F401
import torch  # noqa: F401
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from ..utils.config import config


class FlowerDataset(Dataset):
    """花卉数据集类"""

    def __init__(self, data_dir, transform=None, mode="train"):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.images = []
        self.labels = []
        self.class_to_idx = {}

        self._load_data()

    def _load_data(self):
        """加载数据"""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

        # 获取所有类别
        classes = [
            d
            for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        ]
        classes.sort()
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        # 加载图像和标签
        for class_name in classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.images.append(os.path.join(class_dir, img_file))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class FlowerDataLoader:
    """数据加载器类（无验证集版本）"""

    def __init__(self):
        self.config = config
        self.transform = self._get_transforms()

    def _get_transforms(self):
        """获取数据变换"""
        train_transform = transforms.Compose(
            [
                transforms.Resize(tuple(config.get("data.image_size", [128, 128]))),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize(tuple(config.get("data.image_size", [128, 128]))),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        return {"train": train_transform, "test": test_transform}

    def get_data_loaders(self, data_dir=None, batch_size=None, train_split=None):
        """获取数据加载器（只有训练集和测试集）"""
        if data_dir is None:
            data_dir = config.get("data.raw_data_path", "data/raw")

        if batch_size is None:
            batch_size = config.get("data.batch_size", 32)

        if train_split is None:
            train_split = config.get("data.train_split", 0.8)

        num_workers = config.get("data.num_workers", 4)

        # 创建完整数据集
        full_dataset = FlowerDataset(
            data_dir=data_dir,
            transform=self.transform["train"],  # 使用训练变换
            mode="train",
        )

        # 分割训练集和测试集
        train_size = int(train_split * len(full_dataset))
        test_size = len(full_dataset) - train_size

        train_dataset, test_dataset = random_split(
            full_dataset, [train_size, test_size]
        )

        # 为测试集应用测试变换
        test_dataset.dataset.transform = self.transform["test"]

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        return train_loader, test_loader, full_dataset.class_to_idx

    def get_test_loader(self, test_data_dir=None, batch_size=None):
        """获取独立的测试集加载器"""
        if test_data_dir is None:
            test_data_dir = os.path.join(
                config.get("data.raw_data_path", "data/raw"), "test"
            )

        if batch_size is None:
            batch_size = config.get("data.batch_size", 32)

        num_workers = config.get("data.num_workers", 4)

        # 创建测试数据集
        test_dataset = FlowerDataset(
            data_dir=test_data_dir, transform=self.transform["test"], mode="test"
        )

        # 创建测试数据加载器
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        return test_loader, test_dataset.class_to_idx
