import numpy as np
from PIL import Image, ImageColor

import json

import os
import pickle

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_img(save_path, img_file):
    img = Image.fromarray(img_file)
    ensure_dir(save_path)
    img.save(save_path)

def save_json(save_path, json_file):
    ensure_dir(save_path)
    with open(save_path, 'w') as f:
        json.dump(json_file, f, indent=4)

def save_pkl(save_path,dic_file):
    ensure_dir(save_path)
    with open(save_path, 'wb') as f:
        pickle.dump(dic_file, f)