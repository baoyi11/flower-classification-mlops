import os

import mlflow
import mlflow.pytorch
import numpy as np  # noqa: F401
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (  # noqa: F401
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..utils.config import config
from .cnn_model import create_model


class ModelTrainer:
    """模型训练器（无验证集版本）"""

    def __init__(self, model_name="simple"):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model(model_name).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config.get("model.learning_rate", 0.001)
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.1
        )

        # 设置MLflow
        mlflow.set_tracking_uri(config.get("mlflow.tracking_uri", "mlruns"))
        mlflow.set_experiment(
            config.get("mlflow.experiment_name", "flower-classification")
        )

        # 设置TensorBoard
        self.writer = SummaryWriter("runs/experiment")

    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

            pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)

        # 记录到TensorBoard
        self.writer.add_scalar("Loss/train", epoch_loss, epoch)
        self.writer.add_scalar("Accuracy/train", epoch_acc, epoch)

        return epoch_loss, epoch_acc

    def evaluate(self, test_loader, epoch=None):
        """在测试集上评估模型"""
        self.model.eval()
        test_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(target.cpu().numpy())

        test_loss /= len(test_loader)
        test_acc = accuracy_score(all_labels, all_preds)

        # 记录到TensorBoard
        if epoch is not None:
            self.writer.add_scalar("Loss/test", test_loss, epoch)
            self.writer.add_scalar("Accuracy/test", test_acc, epoch)

        return test_loss, test_acc, all_preds, all_labels

    def train(self, train_loader, test_loader=None, epochs=None):
        """完整训练过程（无验证集）"""
        if epochs is None:
            epochs = config.get("model.epochs", 10)

        best_train_acc = 0.0
        train_losses, train_accs = [], []
        test_losses, test_accs = [], []

        # 开始MLflow运行
        with mlflow.start_run():
            # 记录参数
            mlflow.log_params(
                {
                    "model_type": type(self.model).__name__,
                    "learning_rate": config.get("model.learning_rate", 0.001),
                    "epochs": epochs,
                    "batch_size": config.get("data.batch_size", 32),
                    "image_size": config.get("data.image_size", [128, 128]),
                    "train_split": config.get("data.train_split", 0.8),
                }
            )

            for epoch in range(epochs):
                # 训练
                train_loss, train_acc = self.train_epoch(train_loader, epoch)
                train_losses.append(train_loss)
                train_accs.append(train_acc)

                # 学习率调度
                self.scheduler.step()

                # 记录指标到MLflow
                mlflow.log_metrics(
                    {"train_loss": train_loss, "train_accuracy": train_acc}, step=epoch
                )

                print(f"Epoch {epoch+1}/{epochs}:")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

                # 在测试集上评估（可选）
                if test_loader is not None and epoch % 5 == 0:  # 每5个epoch评估一次
                    test_loss, test_acc, _, _ = self.evaluate(test_loader, epoch)
                    test_losses.append(test_loss)
                    test_accs.append(test_acc)

                    mlflow.log_metrics(
                        {"test_loss": test_loss, "test_accuracy": test_acc}, step=epoch
                    )

                    print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

                # 保存最佳模型（基于训练准确率）
                if train_acc > best_train_acc:
                    best_train_acc = train_acc
                    self.save_model("best_model.pth")
                    mlflow.pytorch.log_model(self.model, "best_model")

            # 最终在测试集上评估
            if test_loader is not None:
                final_test_loss, final_test_acc, final_preds, final_labels = (
                    self.evaluate(test_loader)
                )
                mlflow.log_metrics(
                    {
                        "final_test_loss": final_test_loss,
                        "final_test_accuracy": final_test_acc,
                    }
                )
                print("Final Test Results:")
                print(
                    f"Test Loss: {final_test_loss:.4f}, Test Acc: {final_test_acc:.4f}"
                )

            # 保存最终模型
            self.save_model("final_model.pth")
            mlflow.pytorch.log_model(self.model, "final_model")

            # 记录最佳准确率
            mlflow.log_metric("best_train_accuracy", best_train_acc)

        self.writer.close()

        return {
            "train_losses": train_losses,
            "train_accs": train_accs,
            "test_losses": test_losses,
            "test_accs": test_accs,
            "best_train_acc": best_train_acc,
        }

    def save_model(self, filename):
        """保存模型"""
        model_path = os.path.join(config.get("model.save_path", "models"), filename)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            model_path,
        )

    def load_model(self, filename):
        """加载模型"""
        model_path = os.path.join(config.get("model.save_path", "models"), filename)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
