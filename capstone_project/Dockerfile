# Sử dụng image Ubuntu 22.04
FROM ubuntu:22.04

# Thiết lập biến môi trường
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Cài đặt Python và các công cụ hệ thống
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    cmake \
    ninja-build \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    zlib1g-dev \
    pkg-config \
    libblas-dev \
    libatlas-base-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Đặt thư mục làm việc
WORKDIR /app

# Cài đặt thư viện Python
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn
COPY . .

# Thu thập file tĩnh
RUN python3 manage.py collectstatic --noinput

# Mở cổng
EXPOSE 8000

# Chạy server với Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "capstone_project.wsgi:application"]