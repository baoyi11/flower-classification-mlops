#!/usr/bin/env python3
"""
花卉分类模型训练脚本（无验证集版本）
"""

import os
import argparse
import mlflow
from src.data.data_loader import FlowerDataLoader
from src.model.train import ModelTrainer
from src.utils.config import config

def main():
    parser = argparse.ArgumentParser(description='训练花卉分类模型（无验证集）')
    parser.add_argument('--data_dir', type=str, default=None, help='数据目录路径')
    parser.add_argument('--model', type=str, default='simple', choices=['simple', 'improved'], help='模型类型')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--train_split', type=float, default=None, help='训练集分割比例')
    parser.add_argument('--no_test_eval', action='store_true', help='不在训练过程中进行测试集评估')
    
    args = parser.parse_args()
    
    # 更新配置
    if args.data_dir:
        config.update({'data.raw_data_path': args.data_dir})
    if args.epochs:
        config.update({'model.epochs': args.epochs})
    if args.batch_size:
        config.update({'data.batch_size': args.batch_size})
    if args.lr:
        config.update({'model.learning_rate': args.lr})
    if args.train_split:
        config.update({'data.train_split': args.train_split})
    
    print("开始训练花卉分类模型（无验证集）...")
    print(f"设备: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"数据目录: {config.get('data.raw_data_path')}")
    print(f"模型类型: {args.model}")
    print(f"训练轮数: {config.get('model.epochs')}")
    print(f"批次大小: {config.get('data.batch_size')}")
    print(f"学习率: {config.get('model.learning_rate')}")
    print(f"训练集比例: {config.get('data.train_split')}")
    
    # 加载数据
    print("加载数据...")
    data_loader = FlowerDataLoader()
    train_loader, test_loader, class_to_idx = data_loader.get_data_loaders()
    
    print(f"类别数量: {len(class_to_idx)}")
    print(f"训练批次: {len(train_loader)}")
    print(f"测试批次: {len(test_loader)}")
    
    # 训练模型
    print("开始训练模型...")
    trainer = ModelTrainer(model_name=args.model)
    
    # 根据参数决定是否在训练中使用测试集评估
    test_loader_for_training = None if args.no_test_eval else test_loader
    
    history = trainer.train(train_loader, test_loader_for_training)
    
    print(f"训练完成! 最佳训练准确率: {history['best_train_acc']:.4f}")
    
    # 最终在测试集上评估
    if not args.no_test_eval:
        test_loss, test_acc, _, _ = trainer.evaluate(test_loader)
        print(f"最终测试集准确率: {test_acc:.4f}")
    
    # 记录类别映射
    import json
    with open('artifacts/class_to_idx.json', 'w') as f:
        json.dump(class_to_idx, f, indent=2)
    
    mlflow.log_artifact('artifacts/class_to_idx.json')

if __name__ == '__main__':
    import torch
    main()