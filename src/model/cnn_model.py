import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.config import config


class SimpleCNN(nn.Module):
    """简单的CNN模型"""

    def __init__(self, num_classes=None):
        super(SimpleCNN, self).__init__()

        if num_classes is None:
            num_classes = config.get("model.num_classes", 17)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # 计算全连接层输入尺寸
        image_size = config.get("data.image_size", [128, 128])
        fc_input_size = self._calculate_fc_input_size(image_size)

        self.fc1 = nn.Linear(fc_input_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def _calculate_fc_input_size(self, image_size):
        """计算全连接层输入尺寸"""
        # 模拟前向传播计算尺寸
        x = torch.zeros(1, 3, image_size[0], image_size[1])
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x.numel()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)  # 展平

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class ImprovedCNN(nn.Module):
    """改进的CNN模型"""

    def __init__(self, num_classes=None):
        super(ImprovedCNN, self).__init__()

        if num_classes is None:
            num_classes = config.get("model.num_classes", 17)

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 计算全连接层输入尺寸
        image_size = config.get("data.image_size", [128, 128])
        fc_input_size = self._calculate_fc_input_size(image_size)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fc_input_size, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def _calculate_fc_input_size(self, image_size):
        """计算全连接层输入尺寸"""
        x = torch.zeros(1, 3, image_size[0], image_size[1])
        x = self.features(x)
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_model(model_name="simple", num_classes=None):
    """创建模型工厂函数"""
    if num_classes is None:
        num_classes = config.get("model.num_classes", 17)

    if model_name == "simple":
        return SimpleCNN(num_classes)
    elif model_name == "improved":
        return ImprovedCNN(num_classes)
    else:
        raise ValueError(f"未知的模型类型: {model_name}")
