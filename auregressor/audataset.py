from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from auregressor.autransform import ToTensor, Rescale, Normalize

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class ActionUnitDataset(Dataset):
    """Action Unit dataset."""

    def __init__(self,
                 csv_dir,
                 img_dir,
                 transform=transforms.Compose([
                    Rescale(224),
                    ToTensor(),
                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])):

        self.csv_dir = csv_dir
        self.img_dir = img_dir
        self.transform = transform
        self.columns = [" AU01_r", " AU02_r", " AU04_r", " AU05_r", " AU06_r", " AU07_r",
                        " AU09_r", " AU10_r", " AU12_r", " AU14_r", " AU15_r", " AU17_r",
                        " AU20_r", " AU23_r", " AU25_r", " AU26_r", " AU45_r"]

        self.filenames = [filename[:-4] for filename in os.listdir(self.csv_dir)]
        self.n_images = len(self.filenames)

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.filenames[idx]

        img_path = os.path.join(self.img_dir, filename + ".jpg")
        image = io.imread(img_path)

        csv_path = os.path.join(self.csv_dir, filename + ".csv")
        aus = pd.read_csv(csv_path)[self.columns].to_numpy()[0].reshape(1, -1)

        sample = {'image': image, 'action_units': aus}

        if self.transform:
            sample = self.transform(sample)

        return sample
