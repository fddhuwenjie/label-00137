# NumpyTorch Dockerfile
# 基于 Python 官方镜像，支持 ARM 和 X86 架构

FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置 Python 环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 安装系统依赖（如果需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 设置 PYTHONPATH
ENV PYTHONPATH=/app

# 暴露端口（用于可能的 API 服务）
EXPOSE 8080

# 默认命令：运行测试
CMD ["python", "-m", "pytest", "tests/", "-v"]
