from numpy.lib.arraypad import pad
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.conv import Conv1d


class TNet(nn.Module):
    def __init__(self, k, numpoints=1024, BN=False):
        super().__init__()
        # TODO Add layers: Convolutional k->64, 64->128, 128->1024 with corresponding batch norms and ReLU
        # TODO Add layers: Linear 1024->512, 512->256, 256->k^2 with corresponding batch norms and ReLU
        self.convlayers = nn.Sequential(
            torch.nn.Conv1d(in_channels=k, out_channels=64, kernel_size=1),
            #torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            #torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1),
            #torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
        )
        self.linearlayers = nn.Sequential(
            torch.nn.Linear(1024,512),
            #torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,256),
            #torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,k*k),
        )


        self.register_buffer('identity', torch.from_numpy(np.eye(k).flatten().astype(np.float32)).view(1, k ** 2))
        self.k = k
        self.numpoints = numpoints

    def forward(self, x):
        b = x.shape
        original_x = x

        # TODO Pass input through layers, applying the same max operation as in PointNetEncoder
        # TODO No batch norm and relu after the last Linear layer
        x = self.convlayers(x)
        x = F.max_pool1d(x,kernel_size=self.numpoints)

        
        x = self.linearlayers(x.view(b[0],1024))
        # Adding the identity to constrain the feature transformation matrix to be close to orthogonal matrix
        identity = self.identity.repeat(x.shape[0], 1)
        x = x + identity
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, return_point_features=False, numpoints=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(3,64,kernel_size=1),
            #torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            nn.Conv1d(64,64,kernel_size=1),
            #torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64,128,kernel_size=1),
            #torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            nn.Conv1d(128,1024,kernel_size=1),
            #torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
        )

        self.input_transform_net = TNet(k=3, numpoints=numpoints)
        self.feature_transform_net = TNet(k=64, numpoints=numpoints)

        self.return_point_features = return_point_features

    def forward(self, x):
        num_points = x.shape[2]

        input_transform = self.input_transform_net(x)
        x = torch.bmm(x.transpose(2, 1), input_transform).transpose(2, 1)
        

        x = self.conv1(x)

        feature_transform = self.feature_transform_net(x)
        x = torch.bmm(x.transpose(2, 1), feature_transform).transpose(2, 1)
        point_features = x

        # TODO: Layers 2 and 3: 64->128, 128->1024
        x = self.conv2(x)

        # This is the symmetric max operation
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        if self.return_point_features:
            x = x.view(-1, 1024, 1).repeat(1, 1, num_points)
            return torch.cat([x, point_features], dim=1)
        else:
            return x


            

# class PointNetDecoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv1d(1088,512,kernel_size=1),
#             #torch.nn.BatchNorm1d(512),
#             torch.nn.ReLU(),
#             nn.Conv1d(512,256,kernel_size=1),
#             #torch.nn.BatchNorm1d(256),
#             torch.nn.ReLU(),
#             nn.Conv1d(256,128,kernel_size=1),
#             #torch.nn.BatchNorm1d(128),
#             torch.nn.ReLU(),
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv1d(128,128,kernel_size=1),
#             #torch.nn.BatchNorm1d(128),
#             torch.nn.ReLU(),
#             nn.Conv1d(128,3,kernel_size=1)
#         )
#         # TODO: Define convolutions, batch norms, and ReLU

#     def forward(self, x):
#         # TODO: Pass x through all layers, no batch norm or ReLU after the last conv layer
#         x = self.conv2(self.conv1(x))
#         x = x.transpose(2, 1).contiguous()
#         return x

class PointNetDecoder(nn.Module):
    def __init__(self, numpoints=1024):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1024,1024),
            #torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            nn.Linear(1024,1024),
            #torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            nn.Linear(1024,numpoints*3)
        )
        self.numpoints = numpoints
    
    def forward(self, x):
        bs = x.shape[0]
        x = self.fc(x)
        x = x.view(bs,3,self.numpoints)
        return x








   