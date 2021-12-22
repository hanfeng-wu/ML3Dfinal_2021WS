"""
Mock Dataloader class to use for temp training process
"""

import os
import torch

class ImageDataLoader():

    def __init__(self,length = 10,image_size= 32,out_size = 32):
        self.length = length
        self.in_size =image_size
        self.out_size = out_size

        self.images = torch.rand(length, 3,self.in_size,self.in_size)
        self.labels = torch.rand(length, 1,self.out_size,self.out_size,self.out_size)
        self.labels [self.labels <0.5] = 0.0
        self.labels [self.labels >=0.5] = 1.0


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        
        image = self.images[index]
        label = self.labels[index]

        
        return image,label.long()
