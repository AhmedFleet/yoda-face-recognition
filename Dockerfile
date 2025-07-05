FROM python:3.10-slim

# تثبيت git و OpenCV dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# تعيين مجلد العمل
WORKDIR /app

# نسخ كل المشروع إلى داخل الحاوية
COPY . /app

# تحديث pip وتثبيت المكتبات
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# أمر التشغيل
CMD ["streamlit", "run", "aio.py", "--server.port=8080", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
