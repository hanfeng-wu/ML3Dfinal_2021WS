import torch
from pathlib import Path
import json
import numpy as np
import trimesh
from trimesh import points
import os
import random

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
            self.subdirs = subdirs[0:1920]
        if split == 'val':
            self.subdirs = subdirs[1920:2000]
        if split == 'overfit':
            self.subdirs = subdirs[0:80]    


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

class pcd_stage2(torch.utils.data.Dataset):


    def __init__(self, split, device='cuda'):
        assert split in ['train', 'val', 'overfit']
        self.device = device
        datadir = './Data/shapenetdata/'
        subdirs = []
        pcds = []
        for x in os.walk(datadir):
            subdirs.append(x[0])
        subdirs = sorted(subdirs)[1:]    
        for subdir in subdirs:
            pcds.extend([{'partial': f'{subdir}/pcd_{index}.obj', 'full': f'{subdir}/pcd_gt.obj'}  for index in range(10)])
        
        random.Random(1).shuffle(pcds)
        if split == 'train':
            self.pcds = pcds[0:19000]
        if split == 'val':
            self.pcds = pcds[19000:20000]
        if split == 'overfit':
            self.pcds = pcds[0:10]    


    def __getitem__(self, index):
        partial = self.pcds[index]['partial']
        file_partial = open(partial, 'r')
        obj_partial = trimesh.exchange.obj.load_obj(file_partial)
        full = self.pcds[index]['full']
        file_full = open(full, 'r')
        obj_full = trimesh.exchange.obj.load_obj(file_full)
        return {
            'partial': torch.tensor(obj_partial['vertices']).float(),
            'full': torch.tensor(obj_full['vertices']).float()
        }

    def __len__(self):
        """
        :return: length of the dataset
        """
        # TODO Implement
        return len(self.pcds)

class retrieval_dataset(torch.utils.data.Dataset):
    def __init__(self, split, device='cuda'):
        assert split in ['train', 'val', 'overfit']
        self.device = device
        datadir = './Data/shapenetdata/'
        subdirs = []
        pcds = []
        gts = []
        for x in os.walk(datadir):
            subdirs.append(x[0])
        subdirs = sorted(subdirs)[1:]    
        for subdir in subdirs:
            pcds.extend([{'partial': f'{subdir}/pcd_{index}.obj', 'full': f'{subdir}/pcd_gt.obj'}  for index in range(10)])
            file_full = open(f'{subdir}/pcd_gt.obj', 'r')
            obj_full = trimesh.exchange.obj.load_obj(file_full)
            gts.append({'path': f'{subdir}/pcd_gt.obj', 'tensor': torch.tensor(obj_full['vertices']).float()})
        self.gts = gts    

            
        
        random.Random(1).shuffle(pcds)
        if split == 'train':
            self.pcds = pcds[0:19000]
        if split == 'val':
            self.pcds = pcds[19000:20000]
        if split == 'overfit':
            self.pcds = pcds[0:10]
            
    def __getitem__(self, index):
        partial = self.pcds[index]['partial']
        file_partial = open(partial, 'r')
        obj_partial = trimesh.exchange.obj.load_obj(file_partial)
        return {
            'partial': torch.tensor(obj_partial['vertices']).float(),
            'full': self.pcds[index]['full']
        }
    def __len__(self):
        """
        :return: length of the dataset
        """
        # TODO Implement
        return len(self.pcds)        

    def get_gts(self):
        return self.gts    

    
