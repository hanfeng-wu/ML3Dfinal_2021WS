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
            self.conv2D_1 = nn.Conv3d(10, 64, 3,stride = 4, padding =1) # 32
            torch.nn.init.xavier_uniform_(self.conv2D_1.weight)

            self.conv2D_2 = nn.Conv3d(64, 128, 3, stride = 2, padding =1) # 16
            torch.nn.init.xavier_uniform_(self.conv2D_2.weight)
            
            self.conv2D_3 = nn.Conv3d(128, 256, 3, stride = 2, padding =1) # 8
            torch.nn.init.xavier_uniform_(self.conv2D_3.weight)
            
            # self.bn1 = torch.nn.BatchNorm3d(128)

            in_dim = 8
            out_dim = 4
            self.linear  = nn.Linear(in_dim*in_dim*256, out_dim*out_dim*out_dim*128)
            self.linear_dims = (128,out_dim,out_dim,out_dim)
            # self.bn2 = torch.nn.BatchNorm1d(out_dim*out_dim*out_dim*128)


            self.conv3D_1 = nn.Conv3d(128, 128, 3, padding = 1)
            torch.nn.init.xavier_uniform_(self.conv3D_1.weight)

            # self.bn3 =  torch.nn.BatchNorm3d(128)

            self.conv3D_2 = nn.Conv3d(128, 1, 3, padding = 1)
            torch.nn.init.xavier_uniform_(self.conv3D_2.weight)

            self.upsample = nn.Upsample(scale_factor = 2)
            
            self.hidden_activation = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.dropOut = nn.Dropout(p=0.2)
        
        def forward(self, x_in):
            b = x_in.shape[0]
            
            x = self.hidden_activation(self.conv2D_1(x_in))
            # x = self.dropOut(x)
            
            x = self.hidden_activation(self.conv2D_2(x)) 
            # x = self.bn1(x)
           
            x = self.hidden_activation(self.conv2D_3(x))
            # x = self.dropOut(x) 
           
            x = x.view(b,-1)
            
            x = self.hidden_activation(self.linear(x))
            # x = self.dropOut(x)
            # x = self.bn2(x)
            x = x.view(b,*self.linear_dims)          
            

            x = self.upsample(x)
            x = self.hidden_activation(self.conv3D_1(x))
            # x = self.bn3(x)

            x = self.upsample(x)
            x = self.sigmoid(self.conv3D_2(x))
            
            x = x.view(b,16,16,16)

            return x

    def __init__(self):
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