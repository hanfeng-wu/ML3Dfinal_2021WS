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

            latent_dim = 1024

            kernel_size = 4
            stride = 2
            padding = 1
            layer_0_out_dim = self._conv_layer_output_dim(input_dim=voxel_dimension, kernel_size=kernel_size, stride=stride, padding=padding)
            layer_1_out_dim = self._conv_layer_output_dim(input_dim=layer_0_out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
            layer_1_out_dim /= 2
            layer_2_out_dim = self._conv_layer_output_dim(input_dim=layer_1_out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
            layer_3_out_dim = self._conv_layer_output_dim(input_dim=layer_2_out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
            self._conv_encoded_dim = layer_3_out_dim

            # TODO: add first max pool layer if dim is 256

            self._encoder = nn.Sequential(
                torch.nn.Conv3d(in_channels=1, out_channels=8, kernel_size=kernel_size, stride=stride, padding=1),
                torch.nn.BatchNorm3d(8),
                nn.ReLU(),
                torch.nn.Conv3d(in_channels=8, out_channels=16, kernel_size=kernel_size, stride=stride, padding=1),
                torch.nn.BatchNorm3d(16),
                nn.ReLU(),
                nn.MaxPool3d(2),
                torch.nn.Conv3d(in_channels=16, out_channels=32, kernel_size=kernel_size, stride=stride, padding=1),
                torch.nn.BatchNorm3d(32),
                nn.ReLU(),
                torch.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=stride, padding=1),
                torch.nn.BatchNorm3d(64),
                nn.ReLU(),
                
                nn.Flatten(),
                nn.Linear(layer_3_out_dim * layer_3_out_dim * layer_3_out_dim * 64, latent_dim),
                nn.ReLU(),
            )
            self._decoder_fc = nn.Sequential(
                nn.Linear(latent_dim, layer_3_out_dim * layer_3_out_dim * layer_3_out_dim * 64),
                nn.ReLU(),
            )
            self._decoder = nn.Sequential(
                nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=kernel_size, stride=stride, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=kernel_size, stride=stride, padding=1),
                nn.BatchNorm3d(16),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=kernel_size, stride=stride, padding=1),
                nn.BatchNorm3d(8),
                nn.ReLU(),
                nn.ConvTranspose3d(in_channels=8, out_channels=1, kernel_size=kernel_size, stride=stride, padding=1),
                nn.Sigmoid()
            )

        def _conv_layer_output_dim(self, input_dim: int, kernel_size: int, stride: int, padding: int) -> int:
            return int((input_dim + 2 * padding - kernel_size) / stride + 1)
        
        def forward(self, x):
            encoded = self._encoder.forward(x)
            decoded_fc = self._decoder_fc(encoded)
            reshaped = decoded_fc.view(-1, 64, self._conv_encoded_dim, self._conv_encoded_dim, self._conv_encoded_dim)
            return self._decoder.forward(reshaped)

    def weights_init_uniform(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # apply a uniform distribution to the weights and a bias=0
            m.weight.data.uniform_(0.0, 1.0)
            m.bias.data.fill_(0)

    def __init__(self, voxel_dimension: int, train_set, val_set, test_set, device): # Training and logging
        super().__init__()

        self.data = {'train': train_set,
                     'val': val_set,
                     'test': test_set}

        self.model = VoxelAutoencoder.Network(voxel_dimension)
        self.model.apply(VoxelAutoencoder.weights_init_uniform)
        self.current_device = device

    def forward(self, x_in):
        x = self.model(x_in)     
        return x
        
    def general_step(self, batch, batch_idx, mode):
        voxels = batch

        preds = self.forward(voxels)

        # TODO: cross entropy loss
        loss = nn.L1Loss()(preds, voxels)
        
        temp_preds = preds.clone()
        temp_preds[preds<0.5] = 0
        temp_preds[preds>=0.5] = 1
        
        n_correct = (voxels == temp_preds).sum()
        return loss, n_correct

    def general_end(self, outputs, mode):
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        total_correct = torch.stack(
            [x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / len(self.data[mode])
        return avg_loss, acc

    def training_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss}
        return {'loss': loss, 'train_n_correct': n_correct, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss, 'val_n_correct': n_correct}

    def test_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct': n_correct}

    def validation_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "val")
        print("Val-Acc={}".format(acc))
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc}
        return {'val_loss': avg_loss, 'val_acc': acc, 'log': tensorboard_logs}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data['train'], shuffle=True, batch_size=16)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data['val'], batch_size=16)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.data['test'], batch_size=16)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), 0.01)
        return optim

    def getAcc(self, loader=None):
        self.eval()
        self = self.to(self.current_device)

        if not loader:
            loader = self.test_dataloader()

        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.current_device)
            score = self.forward(X)
            score[score<0.5] = 0
            score[score>=0.5] = 1
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc

