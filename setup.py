from setuptools import setup, find_packages

setup(
    name="flower-classification",
    version="1.0.0",
    description="Flower Image Classification with MLOps",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "mlflow>=2.7.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "Pillow>=10.0.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "opencv-python>=4.8.0",
    ],
    python_requires=">=3.8",
)
