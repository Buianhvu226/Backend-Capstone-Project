# Sử dụng image có sẵn dlib
FROM jgoralcz/python-dlib:latest

# Thiết lập biến môi trường
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Cài đặt các công cụ hệ thống bổ sung
RUN apt-get update && apt-get install -y \
    libpq-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Đặt thư mục làm việc
WORKDIR /app

# Cài đặt thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn
COPY . .

# Thu thập file tĩnh
RUN python manage.py collectstatic --noinput

# Mở cổng
EXPOSE 8000

# Chạy server với Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "capstone_project.wsgi:application"]