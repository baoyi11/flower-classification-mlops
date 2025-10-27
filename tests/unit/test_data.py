import os  # noqa: F401

import pytest
import torch  # noqa: F401

from src.data.data_loader import FlowerDataLoader, FlowerDataset
from src.utils.config import config  # noqa: F401


class TestFlowerDataset:
    """测试花卉数据集"""

    def test_dataset_initialization(self, tmp_path):
        """测试数据集初始化"""
        # 创建测试数据目录结构
        data_dir = tmp_path / "test_data"
        class_dir = data_dir / "class1"
        class_dir.mkdir(parents=True)

        # 创建测试图像文件
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="red")
        img.save(class_dir / "test1.jpg")
        img.save(class_dir / "test2.png")

        # 测试数据集初始化
        dataset = FlowerDataset(str(data_dir))
        assert len(dataset) == 2
        assert "class1" in dataset.class_to_idx

    def test_dataset_transform(self, tmp_path):
        """测试数据变换"""
        data_dir = tmp_path / "test_data"
        class_dir = data_dir / "class1"
        class_dir.mkdir(parents=True)

        from PIL import Image

        img = Image.new("RGB", (100, 100), color="red")
        img.save(class_dir / "test1.jpg")

        from torchvision import transforms

        transform = transforms.Compose(
            [transforms.Resize((128, 128)), transforms.ToTensor()]
        )

        dataset = FlowerDataset(str(data_dir), transform=transform)
        image, label = dataset[0]

        assert image.shape == (3, 128, 128)
        assert isinstance(label, int)


class TestFlowerDataLoader:  # 修复类名
    """测试数据加载器（无验证集版本）"""

    def test_data_loader_initialization(self):
        """测试数据加载器初始化"""
        data_loader = FlowerDataLoader()
        assert data_loader is not None
        assert "train" in data_loader.transform
        assert "test" in data_loader.transform
        assert "val" not in data_loader.transform  # 确保没有验证集变换

    def test_get_data_loaders_no_data(self):
        """测试无数据时的情况"""
        data_loader = FlowerDataLoader()
        with pytest.raises(FileNotFoundError):
            data_loader.get_data_loaders(data_dir="/non/existent/path")

    def test_get_data_loaders_split(self, tmp_path):
        """测试数据分割"""
        # 创建测试数据
        data_dir = tmp_path / "test_data"
        for i in range(3):
            class_dir = data_dir / f"class{i}"
            class_dir.mkdir(parents=True)

            from PIL import Image

            for j in range(10):  # 每个类别10张图像
                img = Image.new("RGB", (100, 100), color=(i * 80, i * 80, i * 80))
                img.save(class_dir / f"img{j}.jpg")

        data_loader = FlowerDataLoader()
        train_loader, test_loader, class_to_idx = data_loader.get_data_loaders(
            data_dir=str(data_dir), train_split=0.8
        )

        # 检查数据加载器
        assert train_loader is not None
        assert test_loader is not None
        assert len(class_to_idx) == 3

        # 检查数据分割比例
        total_samples = len(train_loader.dataset) + len(test_loader.dataset)
        expected_train_size = int(0.8 * total_samples)
        assert len(train_loader.dataset) == expected_train_size

    def test_get_test_loader(self, tmp_path):
        """测试独立测试集加载器"""
        # 创建测试目录结构
        test_data_dir = tmp_path / "test_data" / "test"
        for i in range(2):
            class_dir = test_data_dir / f"class{i}"
            class_dir.mkdir(parents=True)

            from PIL import Image

            for j in range(5):
                img = Image.new("RGB", (100, 100), color=(i * 128, i * 128, i * 128))
                img.save(class_dir / f"img{j}.jpg")

        data_loader = FlowerDataLoader()
        test_loader, class_to_idx = data_loader.get_test_loader(
            test_data_dir=str(test_data_dir), batch_size=2
        )

        assert test_loader is not None
        assert len(class_to_idx) == 2
        assert len(test_loader.dataset) == 10  # 2 classes * 5 images
