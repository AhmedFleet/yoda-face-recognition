#!/bin/bash

# تحديد إصدار بايثون المطلوب
echo "python-3.10" > runtime.txt

# تعطيل العزل أثناء البناء لحل مشكلة Pillow
PIP_NO_BUILD_ISOLATION=0 pip install -r requirements.txt
