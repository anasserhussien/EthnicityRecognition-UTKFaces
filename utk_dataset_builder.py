#%%

import os
import shutil
import torch
from torchvision import transforms
from torch.utils.data import random_split
from sklearn.metrics import classification_report

from torch.utils.data import Dataset
from PIL import Image

import pytorch_lightning as pl


#%% CONSTANTS

# Dir of raw images
DATASET_SRC = 'data/UTKFace'
# Where to save destination
UTK_DIST = 'data/utk_races'

# Must sum to 1.0
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

SEED = 101

pl.seed_everything(SEED)

class UTKFacesDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_list = os.listdir(folder_path)
        self.classes = [0,1,2,3]
       
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.folder_path, image_name)
        image = Image.open(image_path).convert('RGB')
        age, gender, race, _ = image_name.split('_')
        age = int(age)
        gender = int(gender)
        race = int(race)
        #label = torch.tensor([age, gender, race], dtype=torch.float32)
        
        return image, race, image_path
    

# Loading UTKFaces
dataset = UTKFacesDataset(DATASET_SRC)

# Ignore label 4 which is the unknown ethnicity
included_labels = []
for idx in range(len(dataset)):
    if dataset[idx][1] != 4:
        included_labels.append(idx)

dataset = torch.utils.data.Subset(dataset, included_labels)
print('Number of instances:', len(included_labels))

# Get splits number of instances
train_size = int(TRAIN_RATIO * len(dataset))
val_size = int(VAL_RATIO * len(dataset))
test_size = len(dataset) - train_size - val_size

# Split the dataset into train, validation, and test sets
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])



#%% Copy each individual image into corresponding split and label directory

os.makedirs(UTK_DIST, exist_ok=True)
for ds, split in zip([train_dataset, val_dataset, test_dataset], ['train', 'val', 'test']):

    os.makedirs(os.path.join(UTK_DIST, split), exist_ok=True)
    for i in range(len(ds)):
        _, label, img_path = ds.__getitem__(i)

        dist = os.path.join(UTK_DIST, split, str(label))
        os.makedirs(dist, exist_ok=True)
        shutil.copy(img_path, dist)
