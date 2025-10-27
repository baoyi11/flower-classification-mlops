import os

import yaml
from dotenv import load_dotenv

load_dotenv()


class Config:
    """配置管理类"""

    def __init__(self, config_path="config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_paths()

    def _load_config(self):
        """加载YAML配置文件"""
        if not os.path.exists(self.config_path):
            return self._get_default_config()

        try:
            # 使用UTF-8编码读取文件
            with open(self.config_path, "r", encoding="utf-8") as file:
                return yaml.safe_load(file)
        except UnicodeDecodeError:
            # 如果UTF-8失败，尝试其他编码
            try:
                with open(self.config_path, "r", encoding="gbk") as file:
                    return yaml.safe_load(file)
            except Exception as e:
                print(f"无法读取配置文件 {self.config_path}: {e}")
                return self._get_default_config()
        except Exception as e:
            print(f"读取配置文件时出错 {self.config_path}: {e}")
            return self._get_default_config()

    def _get_default_config(self):
        """获取默认配置"""
        return {
            "data": {
                "raw_data_path": "data/raw",
                "processed_data_path": "data/processed",
                "image_size": [128, 128],
                "batch_size": 32,
                "num_workers": 4,
                "train_split": 0.8,  # 训练集比例
            },
            "model": {
                "num_classes": 17,
                "learning_rate": 0.001,
                "epochs": 10,
                "save_path": "models",
            },
            "mlflow": {
                "tracking_uri": "mlruns",
                "experiment_name": "flower-classification",
            },
            "dvc": {"remote_url": None, "remote_name": "origin"},
        }

    def _setup_paths(self):
        """设置路径"""
        paths = [
            self.config["data"]["raw_data_path"],
            self.config["data"]["processed_data_path"],
            self.config["model"]["save_path"],
            "mlruns",
            "artifacts",
        ]

        for path in paths:
            os.makedirs(path, exist_ok=True)

    def get(self, key, default=None):
        """获取配置值"""
        keys = key.split(".")
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else default

    def update(self, updates):
        """更新配置"""
        for key, value in updates.items():
            keys = key.split(".")
            config_level = self.config
            for k in keys[:-1]:
                config_level = config_level.setdefault(k, {})
            config_level[keys[-1]] = value


# 全局配置实例
config = Config()
