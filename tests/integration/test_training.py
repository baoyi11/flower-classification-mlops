import pytest  # noqa: F401
import torch

from src.data.data_loader import FlowerDataLoader
from src.model.train import ModelTrainer


class TestModelTraining:
    """测试模型训练（无验证集版本）"""

    def test_trainer_initialization(self):
        """测试训练器初始化"""
        trainer = ModelTrainer(model_name="simple")
        assert trainer is not None
        # 修复：检查device的类型而不是字符串表示
        assert isinstance(trainer.device, torch.device)
        assert trainer.device.type in ["cuda", "cpu"]
        assert isinstance(trainer.model, torch.nn.Module)
        assert isinstance(trainer.criterion, torch.nn.Module)
        assert isinstance(trainer.optimizer, torch.optim.Optimizer)

    def test_trainer_no_validation(self, tmp_path):
        """测试无验证集的训练"""
        # 创建测试数据
        data_dir = tmp_path / "test_data"
        for i in range(2):
            class_dir = data_dir / f"class{i}"
            class_dir.mkdir(parents=True)

            from PIL import Image

            for j in range(5):  # 每个类别5张图像
                img = Image.new("RGB", (100, 100), color=(i * 128, i * 128, i * 128))
                img.save(class_dir / f"img{j}.jpg")

        data_loader = FlowerDataLoader()
        train_loader, test_loader, _ = data_loader.get_data_loaders(
            data_dir=str(data_dir), batch_size=2, train_split=0.8
        )

        trainer = ModelTrainer(model_name="simple")

        # 训练少量epoch进行测试
        history = trainer.train(train_loader, test_loader=None, epochs=2)

        assert "train_losses" in history
        assert "train_accs" in history
        assert len(history["train_losses"]) == 2
        assert len(history["train_accs"]) == 2

    def test_trainer_evaluation(self, tmp_path):
        """测试模型评估"""
        # 创建测试数据
        data_dir = tmp_path / "test_data"
        for i in range(2):
            class_dir = data_dir / f"class{i}"
            class_dir.mkdir(parents=True)

            from PIL import Image

            for j in range(5):  # 每个类别5张图像
                img = Image.new("RGB", (100, 100), color=(i * 128, i * 128, i * 128))
                img.save(class_dir / f"img{j}.jpg")

        data_loader = FlowerDataLoader()
        train_loader, test_loader, _ = data_loader.get_data_loaders(
            data_dir=str(data_dir), batch_size=2, train_split=0.8
        )

        trainer = ModelTrainer(model_name="simple")

        # 测试评估方法
        test_loss, test_acc, preds, labels = trainer.evaluate(test_loader)

        assert isinstance(test_loss, float)
        assert isinstance(test_acc, float)
        assert 0 <= test_acc <= 1
        assert len(preds) == len(test_loader.dataset)
        assert len(labels) == len(test_loader.dataset)
