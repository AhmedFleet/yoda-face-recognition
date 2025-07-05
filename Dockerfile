FROM python:3.10-slim

# تثبيت مكتبات النظام التي يحتاجها OpenCV و sentence-transformers
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# تعيين مجلد العمل
WORKDIR /app

# نسخ جميع الملفات إلى داخل الحاوية
COPY . /app

# تحديث pip أولًا لتفادي مشاكل dependency
RUN pip install --upgrade pip

# تثبيت المتطلبات
RUN pip install --no-cache-dir -r requirements.txt

# إعداد أمر التشغيل
CMD ["streamlit", "run", "aio.py", "--server.port=8080", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
