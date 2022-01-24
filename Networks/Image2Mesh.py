"""
Folder for the Image2Mesh Network Classes
"""
import torch.nn as nn
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np

class Image2Voxel(pl.LightningModule):
    """
    A class that uses pytorch lightning module to train a network to predict a 3D voxel from a 2d image
    """
    class Network(nn.Module): # the network architecture
        def __init__(self,):
            super().__init__()
            self.conv2D_1 = nn.Conv2d(3, 16, 3,stride = 4, padding =1) 
            self.conv2D_2 = nn.Conv2d(16, 32, 3, stride = 4, padding =1) 
            self.conv2D_3 = nn.Conv2d(32, 32, 3, stride = 4, padding =1) 
            
            self.linear  = nn.Linear(2*2*32, 8*8*8)
            self.linear_dims = (8,8,8)

            self.conv2D_4 = nn.Conv2d(8, 16, 3, padding = 1)
            self.conv2D_5 = nn.Conv2d(16, 32, 3, padding = 1)

            self.upsample2 = nn.Upsample(scale_factor = 2)
            self.upsample4 = nn.Upsample(scale_factor = 2)
            
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.dropOut = nn.Dropout(p=0.2)
        
        def forward(self, x_in):
            b = x_in.shape[0]
            x = self.dropOut(self.relu(self.conv2D_1(x_in)))
            x = self.relu(self.conv2D_2(x)) 
            x = self.dropOut(self.relu(self.conv2D_3(x))) 

            x = x.view(b,-1)
            x = self.dropOut(self.relu(self.linear(x)))
            x = x.view(b,*self.linear_dims)          


            x = self.upsample4(x)
            x = self.relu(self.conv2D_4(x))

            x = self.upsample2(x)
            x = self.sigmoid(self.conv2D_5(x))
            
            return x

    def __init__(self): # Training and logging
        super().__init__()
        self.model = Image2Voxel.Network()
        torch.cuda.empty_cache()

    def forward(self, x_in):
        x = self.model(x_in)     
        return x
        
    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".
        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)