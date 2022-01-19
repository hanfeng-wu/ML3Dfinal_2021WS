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

            latent_dim = int(1024)

            kernel_size = 4
            stride = 2
            padding = 1

            layer_0_out_dim = self._conv_layer_output_dim(input_dim=voxel_dimension, kernel_size=6, stride=4, padding=1)
            layer_0_out_dim /= 2
            layer_1_out_dim = self._conv_layer_output_dim(input_dim=layer_0_out_dim, kernel_size=4, stride=2, padding=1)
            layer_1_out_dim /= 2
            layer_2_out_dim = self._conv_layer_output_dim(input_dim=layer_1_out_dim, kernel_size=3, stride=1, padding=1)
            layer_3_out_dim = self._conv_layer_output_dim(input_dim=layer_2_out_dim, kernel_size=3, stride=1, padding=1)
            layer_3_out_dim /= 2

            self._conv_encoded_dim = int(layer_3_out_dim)


            # TODO: add first max pool layer if dim is 256
            fc_features = int(layer_3_out_dim * layer_3_out_dim * layer_3_out_dim * 256)

            self._encoder = nn.Sequential(
                torch.nn.Conv3d(in_channels=1, out_channels=32, kernel_size=6, stride=4, padding=1),
                torch.nn.BatchNorm3d(32),
                nn.ReLU(),

                nn.MaxPool3d(kernel_size=2),

                torch.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
                torch.nn.BatchNorm3d(64),
                nn.ReLU(),

                nn.MaxPool3d(kernel_size=2),

                torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm3d(128),
                nn.ReLU(),

                torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm3d(256),
                nn.ReLU(),

                nn.MaxPool3d(kernel_size=2),
                
                nn.Flatten(),
                nn.Linear(fc_features, fc_features),
                nn.BatchNorm1d(fc_features),
                nn.ReLU(),
                nn.Linear(fc_features, latent_dim),
                nn.BatchNorm1d(latent_dim),
                nn.ReLU(),
            )
            self._decoder_fc = nn.Sequential(
                nn.Linear(latent_dim, fc_features),
                nn.BatchNorm1d(fc_features),
                nn.ReLU(),
                nn.Linear(fc_features, fc_features),
                nn.BatchNorm1d(fc_features),
                nn.ReLU(),
            )
            self._decoder = nn.Sequential(
                torch.nn.ConvTranspose3d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
                torch.nn.BatchNorm3d(256),
                nn.ReLU(),

                torch.nn.Conv3d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm3d(128),
                nn.ReLU(),

                torch.nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm3d(64),
                nn.ReLU(),

                torch.nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
                torch.nn.BatchNorm3d(64),
                nn.ReLU(),

                nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(),

                torch.nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
                torch.nn.BatchNorm3d(32),
                nn.ReLU(),

                nn.ConvTranspose3d(in_channels=32, out_channels=1, kernel_size=6, stride=4, padding=1),
                nn.BatchNorm3d(1),
                nn.ReLU(),

                nn.ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm3d(1),
                nn.Sigmoid()
            )

        def _conv_layer_output_dim(self, input_dim: int, kernel_size: int, stride: int, padding: int) -> int:
            return int((input_dim + 2 * padding - kernel_size) / stride + 1)
        
        def forward(self, x):
            encoded = self._encoder(x)
            decoded_fc = self._decoder_fc(encoded)
            reshaped = decoded_fc.view(-1, 256, self._conv_encoded_dim, self._conv_encoded_dim, self._conv_encoded_dim)
            return self._decoder(reshaped)

    def __init__(self, voxel_dimension: int, train_set, val_set, test_set, device): # Training and logging
        super().__init__()

        self.data = {'train': train_set,
                     'val': val_set,
                     'test': test_set}

        self.model = VoxelAutoencoder.Network(voxel_dimension)
        self.current_device = device

    def forward(self, x_in):
        x = self.model(x_in)     
        return x
        
    def general_step(self, batch, batch_idx, mode: str):
        x = batch
        y = self(x)
        # TODO: cross entropy loss
        loss = nn.L1Loss()(x, y)
        self.log(f"{mode}_loss", loss, on_step=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "test")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data['train'], shuffle=True, batch_size=16)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data['train'], batch_size=16)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.data['train'], batch_size=16)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), 0.01)
