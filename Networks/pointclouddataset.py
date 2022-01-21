import torch
from pathlib import Path
import json
import numpy as np
import trimesh
from trimesh import points
import os
import random


def pcd_stage2_dataset(device='cuda'):
    # TODO Read sample IDs from the correct split file and store in self.items
    gt = []
    dataset = []
    datadir = './Data/shapenetdata/'
    subdirs = []
    for x in os.walk(datadir):
        subdirs.append(x[0])
        subdirs = sorted(subdirs)
        subdirs = subdirs[1:]

    for subdir in subdirs:
        file = open(f'{subdir}/pcd_gt.obj', 'r')
        obj = trimesh.exchange.obj.load_obj(file)
        gt.append(torch.tensor(obj['vertices']).to(device))
        for index in range(10):
            file = open(f'{subdir}/pcd_{index}.obj', 'r')
            obj = trimesh.exchange.obj.load_obj(file)
            dataset.append({'pcd': torch.tensor(obj['vertices']).to(device), 'gt': len(gt)-1})

    return gt, random.shuffle(dataset)


def pcd_stage1_dataset(device='cuda'):
    gt = []
    datadir = './Data/shapenetdata/'
    subdirs = []
    for x in os.walk(datadir):
        subdirs.append(x[0])
    subdirs = sorted(subdirs)
    subdirs = subdirs[1:]

    for subdir in subdirs:
        file = open(f'{subdir}/pcd_gt.obj', 'r')
        obj = trimesh.exchange.obj.load_obj(file)
        gt.append(torch.tensor(obj['vertices']))

    return random.shuffle(gt) 

class pcd_stage1(torch.utils.data.Dataset):


    def __init__(self, split, device='cuda'):
        assert split in ['train', 'val', 'overfit']
        self.device = device
        datadir = './Data/shapenetdata/'
        subdirs = []
        for x in os.walk(datadir):
            subdirs.append(x[0])
        subdirs = sorted(subdirs)[1:]
        if split == 'train':
            self.subdirs = subdirs[0:1800]
        if split == 'val':
            self.subdirs = subdirs[1800:2000]
        if split == 'overfit':
            self.subdirs = subdirs[0:10]    


    def __getitem__(self, index):
        subdir = self.subdirs[index]
        file = open(f'{subdir}/pcd_gt.obj', 'r')
        obj = trimesh.exchange.obj.load_obj(file)
        return torch.tensor(obj['vertices']).float()

    def __len__(self):
        """
        :return: length of the dataset
        """
        # TODO Implement
        return len(self.subdirs)
    
    @staticmethod
    def move_batch_to_device(batch, device):
        """
        Utility method for moving all elements of the batch to a device
        :return: None, modifies batch inplace
        """
        batch['voxel'] = batch['voxel'].to(device)
