from sklearn.decomposition import PCA
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as transforms
from function import *
import pickle

class FeaturePCA:
    def __init__(self, n_components) -> None:
        self.n_components = n_components
        self.pca = PCA(n_components)
    
    def transform(self, features):
        return self.pca.transform(features)

    def fit(self, features):
        self.pca.fit(features)

    def load_pca(self, file_path):
        with open(file_path, 'rb') as f:
            self.pca = pickle.load(f)

    def save_pca(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.pca, f)

