# 花卉图像分类 MLOps 项目（无验证集版本）

一个完整的花卉图像分类机器学习项目，采用 MLOps 最佳实践，使用训练集和测试集的两分法。

## 项目特点

- 🎯 花卉图像分类（17个类别）
- 🔧 完整的 MLOps 流水线
- 📊 MLflow 实验跟踪
- 🐳 Docker 容器化
- ☁️ DVC 数据版本控制
- 🔍 自动化测试和 CI/CD
- 📈 TensorBoard 可视化
- 🚫 无验证集设计，使用训练集/测试集分割

## 项目结构
flower-classification-mlops/
├── src/ # 源代码
│ ├── data/ # 数据加载和处理
│ ├── model/ # 模型定义和训练
│ └── utils/ # 工具函数和配置
├── tests/ # 测试代码
├── config/ # 配置文件
├── data/ # 数据目录
├── models/ # 训练好的模型
├── mlruns/ # MLflow 实验结果
├── artifacts/ # 其他产出物
└── notebooks/ # Jupyter 笔记本