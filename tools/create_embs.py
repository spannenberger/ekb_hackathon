from transformers import ViTFeatureExtractor, ViTModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import os
import pandas as pd 
from PIL import Image
import numpy as np
# from pytorch_metric_learning import losses
label2idx = {0:'teeth', 1:'caries'}
model = '/workspace/source/model'
# root_dir = '../input/'
batch_size = 8
num_labels = 2
device = 'cpu'
device
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-384")
model = ViTModel.from_pretrained(model)
model.to(device)
model.eval()

res = []
base_embs = {}

for label in ['teeth', 'caries']:
    img_path = [i for i in os.listdir(f'cutted_clear_dataset_v2/cutted_images_{label}') if i.endswith('.jpg')]

    for file in tqdm(img_path):
        image = Image.open(f'cutted_clear_dataset_v2/cutted_images_{label}/{file}').convert('RGB')
        image = feature_extractor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            res.append(model(**image).pooler_output.detach().cpu().numpy())
            
    mean_emb = np.mean(np.asarray(res), axis=0)[0]
    base_embs.update({label:mean_emb})
pd.DataFrame(base_embs).to_csv('base_file_teeth_v3_20ep.csv', index=False)