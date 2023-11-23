import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
# from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision import transforms

import random
# import openslide
import h5py

from PIL import Image
from transformers import AutoTokenizer

def train_collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    rep_tensors, labels = zip(*data)
    
    rep_tensors = torch.vstack(rep_tensors)
    labels = torch.vstack(labels)
    # attention_mask = torch.vstack(attention_mask)
    # n_imgs = torch.vstack([x.unsqueeze(0) for x in n_imgs])

    return rep_tensors, labels

def val_collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    th_img, rep_tensors, caps, caplens, allcaps, n_imgs = zip(*data)
    
    th_img = torch.vstack([x.unsqueeze(0) for x in th_img])
    rep_tensors = torch.vstack(rep_tensors)
    caps = torch.vstack([x.unsqueeze(0) for x in caps])
    caplens = torch.vstack([x.unsqueeze(0) for x in caplens])
    allcaps = torch.vstack([x.unsqueeze(0) for x in allcaps])
    n_imgs = torch.vstack([x.unsqueeze(0) for x in n_imgs])

    return th_img, rep_tensors, caps, caplens, allcaps, n_imgs

# Note: Only works with bs=1
class ResnetPlusVitDataset(Dataset):

    def __init__(self, df_path, text_decode_model, dtype='train', th_transform=None, pid_list=None, max_len=128):
        self.df_path = df_path
        df = pd.read_pickle(self.df_path)
        self.dtype=dtype
        if pid_list==None:
            self.df=df[df.dtype==dtype]
        else:
            self.df=df[df.pid.isin(pid_list)]
        self.th_transform = th_transform
        self.max_len = max_len

        self.text_decode_model = text_decode_model
        self.tokenizer = AutoTokenizer.from_pretrained(text_decode_model)
        # self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        reps_path=self.df.iloc[idx]['reps_path']        
        pixel_values = torch.load(reps_path)
        
        n_imgs = pixel_values.shape[0]
        n_imgs = torch.LongTensor([n_imgs])

        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.padding_side = 'right'
        # labels, attention_mask = self.tokenizer(self.df.iloc[idx]['notes'],return_tensors='pt', padding='max_length', max_length=64).values()
        encoded_dict = self.tokenizer.encode_plus(
            self.df.iloc[idx]['new_notes'],
            return_tensors='pt',
            add_special_tokens = True,
            max_length = self.max_len,
            padding='max_length',
            return_attention_mask = True,
            )
        
        labels = encoded_dict['input_ids']
        attention_mask = encoded_dict['attention_mask']

        if self.th_transform is not None:
            th_img = self.th_transform(th_img)
        
        if self.dtype=='train':
            return pixel_values, labels, attention_mask
        else:
            # all_captions=torch.LongTensor([idx_tokens])
            return pixel_values, labels, attention_mask