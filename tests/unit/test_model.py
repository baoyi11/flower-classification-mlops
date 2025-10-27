import os  # noqa: F401

import pytest
import torch

from src.model.cnn_model import ImprovedCNN, SimpleCNN, create_model
from src.utils.config import config  # noqa: F401


class TestCNNModels:  # 修复类名拼写错误
    """测试CNN模型"""

    def test_simple_cnn_creation(self):
        """测试简单CNN创建"""
        model = SimpleCNN(num_classes=10)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

        # 测试前向传播
        x = torch.randn(2, 3, 128, 128)
        output = model(x)
        assert output.shape == (2, 10)

    def test_improved_cnn_creation(self):
        """测试改进CNN创建"""
        model = ImprovedCNN(num_classes=17)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

        # 测试前向传播
        x = torch.randn(2, 3, 128, 128)
        output = model(x)
        assert output.shape == (2, 17)

    def test_create_model_factory(self):
        """测试模型工厂函数"""
        model = create_model("simple", num_classes=5)
        assert isinstance(model, SimpleCNN)
        assert model.fc2.out_features == 5

        model = create_model("improved", num_classes=8)
        assert isinstance(model, ImprovedCNN)
        assert model.classifier[-1].out_features == 8

        with pytest.raises(ValueError):
            create_model("unknown_model")

    def test_model_on_device(self):
        """测试模型在设备上的运行"""
        model = create_model("simple", num_classes=3)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        x = torch.randn(2, 3, 128, 128).to(device)
        output = model(x)

        assert output.device.type == device.type
        assert output.shape == (2, 3)
