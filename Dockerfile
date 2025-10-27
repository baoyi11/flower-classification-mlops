FROM efreidevopschina.azurecr.io/cache/library/python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p data/raw data/processed models mlruns artifacts

# 设置环境变量
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=file:///app/mlruns

# 暴露端口（如果需要）
EXPOSE 5000

# 设置默认命令
CMD ["python", "train.py", "--epochs", "1", "--batch_size", "8", "--model", "simple", "--no_test_eval"]