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
        Voxel variational autoencoder model based on this paper: https://arxiv.org/pdf/1608.04236.pdf
        """

        def __init__(self):
            super().__init__()

            activation = nn.ELU() # tried ReLU, but ELU (as in the paper) give much better results
            self._encoder = nn.Sequential(
                # 32
                torch.nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
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
                nn.Linear(8*8*8*64, 8*8*8),
                nn.BatchNorm1d(8*8*8),
                activation,
            )

            self._fc_mean = nn.Linear(8*8*8, 128)
            self._bn_mean = nn.BatchNorm1d(128)
            self._fc_std_dev = nn.Linear(8*8*8, 128)
            self._bn_std_dev = nn.BatchNorm1d(128)
            # TODO: add L2 regularization

            self._decoder_fc = nn.Sequential(
                nn.Linear(128, 8*8*8),
                nn.Linear(8*8*8, 8*8*8*64),
                nn.BatchNorm1d(8*8*8*64),
                activation,
            )
            self._decoder = nn.Sequential(
                # 8
                torch.nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
                torch.nn.BatchNorm3d(32),
                activation,

                # 16
                torch.nn.Conv3d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm3d(16),
                activation,
                torch.nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1),
                torch.nn.BatchNorm3d(8),
                activation,

                # 32
                torch.nn.Conv3d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            # encode
            encoded = self._encoder(x)
            mean = self._bn_mean(self._fc_mean(encoded))
            log_std_dev = self._bn_std_dev(self._fc_std_dev(encoded))
            std_dev = torch.exp(log_std_dev)
            normal_distribution_0_1 = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(std_dev))
            normal_distribution = torch.distributions.Normal(mean, std_dev)
            latent_vector = normal_distribution.rsample()

            # decode
            decoded_fc = self._decoder_fc(latent_vector)
            reshaped = decoded_fc.view(-1, 64, 8, 8, 8)
            return self._decoder(reshaped), normal_distribution_0_1, normal_distribution

        def encode(self, x):
            encoded = self._encoder(x)
            mean = self._bn_mean(self._fc_mean(encoded))
            log_std_dev = self._bn_std_dev(self._fc_std_dev(encoded))
            std_dev = torch.exp(log_std_dev)
            normal_distribution = torch.distributions.Normal(mean, std_dev)
            latent_vector = normal_distribution.rsample()
            return latent_vector

    def __init__(self, train_set, val_set, test_set, device, kl_divergence_scale = 0.1):
        super().__init__()

        self._data = {'train': train_set,
                     'val': val_set,
                     'test': test_set}

        self._model = VoxelAutoencoder.Network()
        self._device = device
        self._kl_divergence_scale = kl_divergence_scale

    def forward(self, x):
        return self._model(x)[0]   

    def encode(self, x):
        return self._model.encode(x)
        
    def general_step(self, batch, batch_idx, mode: str):
        target = batch
        prediction, normal_distribution_0_1, normal_distribution = self._model(target)

        # Give higher weight to False negatives
        filled_fraction_in_batch = (target.sum() / target.numel()).item()
        # clamp the fraction, otherwise we start to get many false positives
        filled_fraction_in_batch = max(0.03, filled_fraction_in_batch)
        weights = torch.empty(target.shape)
        weights[target < 0.5] = filled_fraction_in_batch
        weights[target >= 0.5] = 1 - filled_fraction_in_batch
        weights = weights.to(self._device)

        reconstruction_loss = nn.BCELoss(reduction="none")(prediction, target)
        reconstruction_loss = (reconstruction_loss * weights).mean()

        kl_divergence = torch.distributions.kl_divergence(normal_distribution, normal_distribution_0_1).mean()

        loss = kl_divergence * self._kl_divergence_scale + reconstruction_loss

        self.log(f"{mode}_reconstruction_loss", reconstruction_loss, on_step=False, on_epoch=True)
        self.log(f"{mode}_kl_divergence", kl_divergence, on_step=False, on_epoch=True)
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
        return torch.utils.data.DataLoader(self._data['train'], shuffle=True, batch_size=64, num_workers=0)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self._data['train'], batch_size=64, num_workers=0)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self._data['train'], batch_size=64, num_workers=0)

    def configure_optimizers(self):
        return torch.optim.Adam(self._model.parameters(), lr=0.001, weight_decay=0.001)
