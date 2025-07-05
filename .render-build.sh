#!/bin/bash
set -e  # أي خطأ يوقف التنفيذ فورًا

echo "🔧 Disabling build isolation for pip..."
export PIP_NO_BUILD_ISOLATION=0

echo "📦 Installing requirements..."
pip install -r requirements.txt
