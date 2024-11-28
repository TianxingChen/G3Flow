#!/bin/bash
pip install openmim -i https://pypi.org/simple
#mim install mmcv-full==1.7.1 -i https://pypi.org/simple
pip install -r requirements.txt -i https://pypi.org/simple
cd transformer_utils && python setup.py install