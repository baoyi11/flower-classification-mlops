#!/usr/bin/env python3
"""
项目验证脚本（无验证集版本）
验证项目结构、配置和基本功能
"""

import os
import sys
import importlib
import subprocess
from pathlib import Path

def check_directory_structure():
    """检查目录结构"""
    required_dirs = [
        'src/data',
        'src/model', 
        'src/utils',
        'tests/unit',
        'tests/integration',
        'data/raw',
        'data/processed',
        'models',
        'mlruns',
        'artifacts',
        'config',
        '.github/workflows'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"❌ 缺少必要的目录: {missing_dirs}")
        return False
    else:
        print("✅ 目录结构完整")
        return True

def check_required_files():
    """检查必要文件"""
    required_files = [
        'requirements.txt',
        'config/config.yaml',
        'src/__init__.py',
        'src/data/__init__.py',
        'src/model/__init__.py',
        'src/utils/__init__.py',
        'train.py',
        'README.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ 缺少必要的文件: {missing_files}")
        return False
    else:
        print("✅ 必要文件完整")
        return True

def check_python_imports():
    """检查Python导入"""
    modules_to_check = [
        'torch',
        'torchvision',
        'mlflow',
        'pandas',
        'numpy',
        'sklearn',
        'PIL',
        'src.utils.config',
        'src.data.data_loader',
        'src.model.cnn_model'
    ]
    
    failed_imports = []
    for module_name in modules_to_check:
        try:
            importlib.import_module(module_name)
            print(f"✅ 成功导入: {module_name}")
        except ImportError as e:
            failed_imports.append(module_name)
            print(f"❌ 导入失败: {module_name} - {e}")
    
    if failed_imports:
        print(f"❌ 导入失败的模块: {failed_imports}")
        return False
    else:
        print("✅ 所有模块导入成功")
        return True

def run_tests():
    """运行测试"""
    try:
        result = subprocess.run([
            'python', '-m', 'pytest', 'tests/', '-v', '--tb=short'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ 所有测试通过")
            return True
        else:
            print(f"❌ 测试失败:\n{result.stdout}\n{result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ 测试超时")
        return False
    except Exception as e:
        print(f"❌ 运行测试时出错: {e}")
        return False

def check_code_quality():
    """检查代码质量"""
    try:
        # 检查代码格式
        result_black = subprocess.run([
            'python', '-m', 'black', '--check', 'src/', 'tests/'
        ], capture_output=True, text=True)
        
        # 检查导入排序
        result_isort = subprocess.run([
            'python', '-m', 'isort', '--check-only', 'src/', 'tests/'
        ], capture_output=True, text=True)
        
        # 检查代码规范
        result_flake8 = subprocess.run([
            'python', '-m', 'flake8', 'src/', 'tests/'
        ], capture_output=True, text=True)
        
        if result_black.returncode == 0:
            print("✅ 代码格式检查通过 (Black)")
        else:
            print(f"❌ 代码格式检查失败 (Black):\n{result_black.stdout}")
        
        if result_isort.returncode == 0:
            print("✅ 导入排序检查通过 (isort)")
        else:
            print(f"❌ 导入排序检查失败 (isort):\n{result_isort.stdout}")
        
        if result_flake8.returncode == 0:
            print("✅ 代码规范检查通过 (flake8)")
        else:
            print(f"❌ 代码规范检查失败 (flake8):\n{result_flake8.stdout}")
        
        return all([
            result_black.returncode == 0,
            result_isort.returncode == 0,
            result_flake8.returncode == 0
        ])
        
    except Exception as e:
        print(f"❌ 代码质量检查时出错: {e}")
        return False

def check_no_validation_set():
    """检查无验证集设计"""
    try:
        # 检查数据加载器
        from src.data.data_loader import FlowerDataLoader
        data_loader = FlowerDataLoader()
        
        # 检查变换字典
        transforms = data_loader.transform
        if 'val' in transforms:
            print("❌ 数据变换中包含验证集变换")
            return False
        
        # 检查数据加载方法
        train_loader, test_loader, class_to_idx = data_loader.get_data_loaders()
        
        if train_loader is None or test_loader is None:
            print("❌ 数据加载器返回空值")
            return False
        
        print("✅ 无验证集设计正确实现")
        return True
        
    except Exception as e:
        print(f"❌ 检查无验证集设计时出错: {e}")
        return False

def main():
    """主验证函数"""
    print("开始项目验证（无验证集版本）...\n")
    
    checks = [
        ("目录结构", check_directory_structure),
        ("必要文件", check_required_files),
        ("Python导入", check_python_imports),
        ("无验证集设计", check_no_validation_set),
        ("代码质量", check_code_quality),
        ("测试", run_tests)
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\n--- 检查: {check_name} ---")
        result = check_func()
        results.append((check_name, result))
    
    print("\n" + "="*50)
    print("验证结果汇总:")
    print("="*50)
    
    all_passed = True
    for check_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{check_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 所有验证检查通过! 项目设置完成。")
        return 0
    else:
        print("⚠️  部分验证检查失败，请检查并修复问题。")
        return 1

if __name__ == "__main__":
    sys.exit(main())