# Sử dụng image có sẵn face_recognition + dlib
FROM ageitgey/face_recognition:latest

# Tạo thư mục làm việc
WORKDIR /app

# Cài thêm pip, nếu cần
RUN apt-get update && apt-get install -y python3-pip

# Copy code vào
COPY . .

# Cài các thư viện python (trừ dlib, face_recognition)
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://pypi.org/simple \
    --no-deps

# Collect static nếu dùng
RUN python3 manage.py collectstatic --noinput || true

# Chạy bằng gunicorn (đổi tên project của bạn)
CMD ["gunicorn", "your_project.wsgi:application", "--bind", "0.0.0.0:8000"]
