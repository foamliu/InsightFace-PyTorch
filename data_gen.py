import os
import pickle

import cv2 as cv
import numpy as np
from torch.utils.data import Dataset

from config import IMG_DIR
from config import pickle_file


class ArcFaceDataset(Dataset):
    def __init__(self, split):
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)

        self.split = split
        self.samples = data

    def __getitem__(self, i):
        sample = self.samples[i]
        filename = sample['img']
        label = sample['label']

        filename = os.path.join(IMG_DIR, filename)
        img = cv.imread(filename)
        img = ((img - 127.5) / 128.).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))  # HxWxC array to CxHxW

        return img, label

    def __len__(self):
        return len(self.samples)
