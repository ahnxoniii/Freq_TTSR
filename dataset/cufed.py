import os
from imageio import imread
from PIL import Image
import numpy as np
import glob
import random
import logging

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from fusion_utils.saliency import Saliency
from pathlib import Path

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def check_mask(root: Path, img_list):
    mask_cache = True
    if (root / 'mask').exists():
        for img_name in img_list:
            if not (root / 'mask' / img_name).exists():
                mask_cache = False
                break
    else:
        mask_cache = False
    if mask_cache:
        logging.info('find mask cache in folder, skip saliency detection')
    else:
        logging.info('find no mask cache in folder, start saliency detection')
        saliency = Saliency()
        saliency.inference(src=root / 'input', dst=root / 'mask', suffix='png')


class RandomRotate(object):
    def __call__(self, sample):
        k1 = np.random.randint(0, 4)
        sample['LR'] = np.rot90(sample['LR'], k1).copy()
        sample['HR'] = np.rot90(sample['HR'], k1).copy()
        sample['LR_sr'] = np.rot90(sample['LR_sr'], k1).copy()
        #sample['mask'] = np.rot90(sample['mask'], k1).copy()
        k2 = np.random.randint(0, 4)
        sample['Ref'] = np.rot90(sample['Ref'], k2).copy()
        sample['Ref_sr'] = np.rot90(sample['Ref_sr'], k2).copy()
        return sample


class RandomFlip(object):
    def __call__(self, sample):
        if (np.random.randint(0, 2) == 1):
            sample['LR'] = np.fliplr(sample['LR']).copy()
            sample['HR'] = np.fliplr(sample['HR']).copy()
            sample['LR_sr'] = np.fliplr(sample['LR_sr']).copy()
            #sample['mask'] = np.fliplr(sample['mask']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['Ref'] = np.fliplr(sample['Ref']).copy()
            sample['Ref_sr'] = np.fliplr(sample['Ref_sr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['LR'] = np.flipud(sample['LR']).copy()
            sample['HR'] = np.flipud(sample['HR']).copy()
            sample['LR_sr'] = np.flipud(sample['LR_sr']).copy()
            #sample['mask'] = np.flipud(sample['mask']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['Ref'] = np.flipud(sample['Ref']).copy()
            sample['Ref_sr'] = np.flipud(sample['Ref_sr']).copy()
        return sample


class ToTensor(object):
    def __call__(self, sample):
        LR, LR_sr, HR, Ref, Ref_sr = sample['LR'], sample['LR_sr'], sample['HR'], sample['Ref'], sample['Ref_sr']
        LR = LR.transpose((2,0,1))
        LR_sr = LR_sr.transpose((2,0,1))
        HR = HR.transpose((2,0,1))
        Ref = Ref.transpose((2,0,1))
        Ref_sr = Ref_sr.transpose((2,0,1))
        # mask = np.expand_dims(mask, axis=-1)
        # mask = mask.transpose((2,0,1))
        return {'LR': torch.from_numpy(LR).float(),
                'LR_sr': torch.from_numpy(LR_sr).float(),
                'HR': torch.from_numpy(HR).float(),
                'Ref': torch.from_numpy(Ref).float(),
                'Ref_sr': torch.from_numpy(Ref_sr).float(),
                #'mask': torch.from_numpy(mask).float()
                }


class TrainSet(Dataset):
    def __init__(self, args, transform=transforms.Compose([RandomFlip(), RandomRotate(), ToTensor()]) ):
        self.input_list = sorted([os.path.join(args.dataset_dir, 'train/input', name) for name in 
            os.listdir( os.path.join(args.dataset_dir, 'train/input') )])
        self.img_list = sorted([ name for name in os.listdir(os.path.join(args.dataset_dir, 'train/input'))])
        self.ref_list = sorted([os.path.join(args.dataset_dir, 'train/ref', name) for name in 
            os.listdir( os.path.join(args.dataset_dir, 'train/ref') )])
        # self.mask_list = sorted([os.path.join(args.dataset_dir, 'train/mask', name) for name in 
        #     os.listdir( os.path.join(args.dataset_dir, 'train/mask') )])
        self.transform = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        ### HR
        HR = imread(self.input_list[idx])
        h,w = HR.shape[:2]

        ### mask
        #check_mask(Path("/home/soeunan/TTSR_fre/dataset/CUFED/train"), self.img_list)
        #mask = imread(self.mask_list[idx])
        #print("mask shape", mask.shape)

        ### LR and LR_sr
        LR = np.array(Image.fromarray(HR).resize((w//4, h//4), Image.BICUBIC))
        LR_sr = np.array(Image.fromarray(LR).resize((w, h), Image.BICUBIC))

        ### Ref and Ref_sr
        Ref_sub = imread(self.ref_list[idx])
        h2, w2 = Ref_sub.shape[:2]
        Ref_sr_sub = np.array(Image.fromarray(Ref_sub).resize((w2//4, h2//4), Image.BICUBIC))
        Ref_sr_sub = np.array(Image.fromarray(Ref_sr_sub).resize((w2, h2), Image.BICUBIC))
    
        ### complete ref and ref_sr to the same size, to use batch_size > 1
        Ref = np.zeros((160, 160, 3))
        Ref_sr = np.zeros((160, 160, 3))
        Ref[:h2, :w2, :] = Ref_sub
        Ref_sr[:h2, :w2, :] = Ref_sr_sub

        ### change type
        LR = LR.astype(np.float32)
        LR_sr = LR_sr.astype(np.float32)
        HR = HR.astype(np.float32)
        Ref = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)
       #mask = mask.astype(np.float32)

        ### rgb range to [-1, 1]
        LR = LR / 127.5 - 1.
        LR_sr = LR_sr / 127.5 - 1.
        HR = HR / 127.5 - 1.
        Ref = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.
        #mask = 2 * mask - 1

        sample = {'LR': LR,  
                  'LR_sr': LR_sr,
                  'HR': HR,
                  'Ref': Ref, 
                  'Ref_sr': Ref_sr,
                  #'mask': mask
                  }

        if self.transform:
            sample = self.transform(sample)
        return sample



class TestSet(Dataset):
    def __init__(self, args, ref_level='1', transform=transforms.Compose([ToTensor()])):
        self.input_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'test/CUFED5', '*_0.png')))
        self.ref_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'test/CUFED5', 
            '*_' + ref_level + '.png')))
        self.transform = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        ### HR
        HR = imread(self.input_list[idx])
        h, w = HR.shape[:2]
        #h, w = h//32*32, w//32*32
        h, w = h//8*8, w//8*8
        HR = HR[:h, :w, :] ### crop to the multiple of 4

        ### LR and LR_sr
        LR = np.array(Image.fromarray(HR).resize((w//4, h//4), Image.BICUBIC))
        LR_sr = np.array(Image.fromarray(LR).resize((w, h), Image.BICUBIC))

        ### Ref and Ref_sr
        Ref = imread(self.ref_list[idx])
        h2, w2 = Ref.shape[:2]
        # h2, w2 = h2//32*32, w2//32*32
        h2, w2 = h2//8*8, w2//8*8
        
        Ref = Ref[:h2, :w2, :]
        Ref_sr = np.array(Image.fromarray(Ref).resize((w2//4, h2//4), Image.BICUBIC))
        Ref_sr = np.array(Image.fromarray(Ref_sr).resize((w2, h2), Image.BICUBIC))

        ### change type
        LR = LR.astype(np.float32)
        LR_sr = LR_sr.astype(np.float32)
        HR = HR.astype(np.float32)
        Ref = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        ### rgb range to [-1, 1]
        LR = LR / 127.5 - 1.
        LR_sr = LR_sr / 127.5 - 1.
        HR = HR / 127.5 - 1.
        Ref = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.

        sample = {'LR': LR,  
                  'LR_sr': LR_sr,
                  'HR': HR,
                  'Ref': Ref, 
                  'Ref_sr': Ref_sr}

        if self.transform:
            sample = self.transform(sample)
        return sample