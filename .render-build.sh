#!/bin/bash
echo "python-3.10.13" > runtime.txt
export PIP_NO_BUILD_ISOLATION=0
pip install -r requirements.txt
