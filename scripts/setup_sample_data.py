#!/usr/bin/env python3
"""
设置样本数据脚本（无验证集版本）
创建小规模的样本数据用于测试
"""

import os  # noqa: F401
import shutil  # noqa: F401
from pathlib import Path

import numpy as np
from PIL import Image


def create_sample_data():
    """创建样本数据（无验证集版本）"""
    base_dir = Path("data/raw")

    # 创建目录结构 - 只有类别目录，没有train/val/test子目录
    classes = ["daisy", "rose", "sunflower", "tulip", "dandelion"]

    for class_name in classes:
        (base_dir / class_name).mkdir(parents=True, exist_ok=True)

    # 创建样本图像
    print("创建样本图像...")
    for class_idx, class_name in enumerate(classes):
        for i in range(10):  # 每个类别创建10张图像
            # 创建简单的彩色图像
            img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)

            # 保存图像
            img_path = base_dir / class_name / f"sample_{i}.jpg"
            img.save(img_path)

            print(f"创建: {img_path}")

    print("样本数据创建完成!")


def validate_sample_data():
    """验证样本数据"""
    base_dir = Path("data/raw")

    required_classes = ["daisy", "rose", "sunflower", "tulip", "dandelion"]

    for class_name in required_classes:
        class_path = base_dir / class_name
        if not class_path.exists():
            print(f"❌ 缺少类别目录: {class_path}")
            return False

        # 检查目录中是否有图像文件
        image_files = list(class_path.glob("*.jpg"))
        if len(image_files) == 0:
            print(f"❌ 类别目录中没有图像文件: {class_path}")
            return False

    print("✅ 样本数据验证通过")
    return True


if __name__ == "__main__":
    create_sample_data()
    validate_sample_data()
