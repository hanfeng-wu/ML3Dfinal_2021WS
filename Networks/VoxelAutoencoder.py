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
        """
        Voxel autoencoder model based on the architecture in this paper: https://arxiv.org/pdf/1608.04236.pdf
        """

        def __init__(self, voxel_dimension: int):
            """
            voxel_dimension: side length of the input voxels
            """
            super().__init__()

            activation = nn.ELU()

            self._encoder = nn.Sequential(
                torch.nn.Conv3d(in_channels=1, out_channels=2, kernel_size=4, stride=2, padding=1),
                torch.nn.BatchNorm3d(2),
                activation,

                torch.nn.Conv3d(in_channels=2, out_channels=4, kernel_size=4, stride=2, padding=1),
                torch.nn.BatchNorm3d(4),
                activation,

                # 32

                torch.nn.Conv3d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm3d(8),
                activation,

                torch.nn.Conv3d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=1),
                torch.nn.BatchNorm3d(16),
                activation,

                # 16

                torch.nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm3d(32),
                activation,

                torch.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
                torch.nn.BatchNorm3d(64),
                activation,

                # 8
                
                nn.Flatten(),
                # nn.Linear(fc_features, fc_features),
                # nn.BatchNorm1d(fc_features),
                # nn.ReLU(),
                nn.Linear(8*8*8*64, 8*8*8*4),
                nn.BatchNorm1d(8*8*8*4),
                activation,
                nn.Linear(8*8*8*4, 512),
            )
            self._decoder_fc = nn.Sequential(
                nn.Linear(512, 8*8*8*4),
                nn.Linear(8*8*8*4, 8*8*8*64),
                nn.BatchNorm1d(8*8*8*64),
                activation,
            )
            self._decoder = nn.Sequential(
                # 8

                torch.nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
                torch.nn.BatchNorm3d(32),
                activation,

                # 16

                torch.nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm3d(16),
                activation,

                torch.nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1),
                torch.nn.BatchNorm3d(8),
                activation,

                # 32

                torch.nn.ConvTranspose3d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm3d(4),
                activation,

                nn.ConvTranspose3d(in_channels=4, out_channels=2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(2),
                activation,

                nn.ConvTranspose3d(in_channels=2, out_channels=1, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(1),
                nn.Sigmoid()
            )

        def _conv_layer_output_dim(self, input_dim: int, kernel_size: int, stride: int, padding: int) -> int:
            return int((input_dim + 2 * padding - kernel_size) / stride + 1)
        
        def forward(self, x):
            encoded = self._encoder(x)
            decoded_fc = self._decoder_fc(encoded)
            reshaped = decoded_fc.view(-1, 64, 8, 8, 8)
            return self._decoder(reshaped)

    def __init__(self, voxel_dimension: int, train_set, val_set, test_set, device): # Training and logging
        super().__init__()

        self.data = {'train': train_set,
                     'val': val_set,
                     'test': test_set}

        self.model = VoxelAutoencoder.Network(voxel_dimension)
        self._device = device

    def forward(self, x_in):
        x = self.model(x_in)     
        return x
        
    def general_step(self, batch, batch_idx, mode: str):
        target = batch
        prediction = self(target)

        # Give higher weight to False negatives
        filled_fraction_in_batch = (target.sum() / target.numel()).item()
        # clamp the fraction, otherwise we start to get many false positives
        filled_fraction_in_batch = max(0.02, filled_fraction_in_batch)
        weights = torch.empty(target.shape)
        weights[target < 0.5] = filled_fraction_in_batch
        weights[target >= 0.5] = 1 - filled_fraction_in_batch
        weights = weights.to(self._device)

        loss = nn.BCELoss(reduction="none")(prediction, target)
        loss = (loss * weights).mean()

        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True)
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
