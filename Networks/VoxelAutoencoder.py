"""
Folder for the Image2Mesh Network Classes
"""
import torch.nn as nn
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np

class VoxelAutoencoder(pl.LightningModule):
    
    class Network(nn.Module):
        def __init__(self, voxel_dimension: int):
            """
            voxel_dimension: side length of the input voxels
            """
            super().__init__()

            kernel_size = 4
            stride = 3
            padding = 1
            layer_0_out_dim = self._conv_layer_output_dim(input_dim=voxel_dimension, kernel_size=kernel_size, stride=stride, padding=padding)
            layer_1_out_dim = self._conv_layer_output_dim(input_dim=layer_0_out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
            layer_2_out_dim = self._conv_layer_output_dim(input_dim=layer_1_out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
            layer_3_out_dim = self._conv_layer_output_dim(input_dim=layer_2_out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
            self._conv_encoded_dim = layer_3_out_dim

            self._encoder = nn.Sequential(
                torch.nn.Conv3d(in_channels=1, out_channels=8, kernel_size=kernel_size, stride=stride, padding=1),
                torch.nn.BatchNorm3d(8),
                nn.ReLU(),
                torch.nn.Conv3d(in_channels=8, out_channels=16, kernel_size=kernel_size, stride=stride, padding=1),
                torch.nn.BatchNorm3d(16),
                nn.ReLU(),
                torch.nn.Conv3d(in_channels=16, out_channels=32, kernel_size=kernel_size, stride=stride, padding=1),
                torch.nn.BatchNorm3d(32),
                nn.ReLU(),
                torch.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=stride, padding=1),
                torch.nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(layer_3_out_dim * layer_3_out_dim * layer_3_out_dim * 64, 512),
                nn.ReLU(),
            )
            self._decoder_fc = nn.Sequential(
                nn.Linear(512, layer_3_out_dim * layer_3_out_dim * layer_3_out_dim * 64),
                nn.ReLU(),
            )
            self._decoder = nn.Sequential(
                nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=kernel_size, stride=stride, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=kernel_size, stride=stride, padding=1),
                nn.BatchNorm3d(16),
                nn.ReLU(),
                nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=kernel_size, stride=stride, padding=1),
                nn.BatchNorm3d(8),
                nn.ReLU(),
                nn.ConvTranspose3d(in_channels=8, out_channels=1, kernel_size=kernel_size, stride=stride, padding=1),
                nn.ReLU(),
            )

        def _conv_layer_output_dim(self, input_dim: int, kernel_size: int, stride: int, padding: int) -> int:
            return int((input_dim + 2 * padding - kernel_size) / stride + 1)
        
        def forward(self, x):
            encoded = self._encoder.forward(x)
            print(encoded.shape)
            decoded_fc = self._decoder_fc(encoded)
            print(decoded_fc.shape)
            reshaped = decoded_fc.view(-1, 64, self._conv_encoded_dim, self._conv_encoded_dim, self._conv_encoded_dim)
            return self._decoder.forward(reshaped)


