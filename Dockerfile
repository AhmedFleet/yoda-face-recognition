FROM python:3.10-slim

# تثبيت مكتبات النظام التي يحتاجها OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# تعيين مجلد العمل
WORKDIR /app

# نسخ جميع ملفات المشروع
COPY . /app

# تثبيت مكتبات بايثون من requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# أمر التشغيل
CMD ["streamlit", "run", "aio.py", "--server.port=8080", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
