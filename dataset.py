import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
import torch 
from torch import nn


class SharadaDataset(Dataset):
    """Scripture dataset Class."""

    def __init__(self, files_dir, transform=None, char_dict=None, max_len=None, null_annot=None):
        """
        Args:
            files_dir (string): Path to the txt file with labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.files_dir = files_dir
        self.transform = transform
        self.max_len = max_len
        self.char_list = " -ँंःअआइईउऊऋऌऍऎएऐऑऒओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळऴवशषसह़ऽािीुूृॄॅेैॉॊोौ्ॐ॒॑॓॔क़ख़ग़ज़ड़ढ़फ़य़ॠॢ।॥०१२३४५६७८९॰ॱॲॻॼॽॾ≈–|"
        self.null_annot = null_annot
        
        if self.char_list is not None:
            char_filtered = []
            chars = sorted(list(set(self.char_list)))
            for char in chars:
                if char not in self.null_annot:
                    char_filtered.append(char)
            self.char_dict = {c:i for i,c in enumerate(char_filtered)}
        # print(f"Char Dict{self.char_dict}")
        self.idx_to_char = {k:v for v,k in self.char_dict.items()}
        f_ = os.listdir(self.files_dir)
        self.files = list(set([i.split('.')[0] for i in f_]))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx] + '.jpg'
        img_filepath = os.path.join(self.files_dir,img_name)
        try:
            image = Image.open(img_filepath)

        except OSError:
            image = np.random.randint(0, 255, size=(50, 100), dtype=np.uint8)

        txt_name = self.files[idx] + '.txt'
        txt_filepath = os.path.join(self.files_dir,txt_name)
        try:
            with open(txt_filepath,'r') as file:
                label = file.read()

        except OSError:
            label = ""

        # if len(label) > self.max_len:
        #     self.max_len = len(label)
            

        label_ = [self.char_dict.get(letter) for letter in label] # Temporary encoded label
        label_ = torch.tensor(label_)
        # print(label_)
        # print(self.max_len - label_.shape[0])

        padded_label = nn.ConstantPad1d((0, self.max_len - label_.shape[0]), 0)(label_)

        # print(padded_label.shape)

        sample = {'image': image, 'label': padded_label}
        # print(sample)
        if self.transform:
            sample = self.transform(sample)
        return sample