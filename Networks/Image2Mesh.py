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
            self.conv2D_1 = nn.Conv2d(3, 16, 3,stride = 2, padding =1)
 
            self.conv2D_2 = nn.Conv2d(16, 32, 3, stride = 2, padding =1) 

            self.linear  = nn.Linear(8*8*32, 32*32*32)

            self.conv3D_1 = nn.Conv3d(1,32,3,padding = 1)
            self.conv3D_2 = nn.Conv3d(32,1,3,padding = 1)
            
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.dropOut = nn.Dropout(p=0.2)
        
        def forward(self, x_in):
            b = x_in.shape[0]
            x = self.dropOut(self.relu(self.conv2D_1(x_in)))
            x = self.dropOut(self.relu(self.conv2D_2(x))) 

            x = x.view(b,-1)
            x = self.dropOut(self.relu(self.linear(x)))

            x = x.view(b,1,32,32,32)          

            x = self.dropOut(self.relu(self.conv3D_1(x)))
            x = self.dropOut(self.sigmoid(self.conv3D_2(x)))
            
            return x

    def __init__(self,train_set, val_set, test_set,device): # Training and logging
        super().__init__()

        self.data = {'train': train_set,
                     'val': val_set,
                     'test': test_set}

        self.model = Image2Voxel.Network()
        self.current_device = device

    def forward(self, x_in):
        x = self.model(x_in)     
        return x
        
    def general_step(self, batch, batch_idx, mode):
        images, targets = batch

        preds = self.forward(images)

        loss = nn.L1Loss()(preds, targets)
        
        temp_preds = preds.clone()
        temp_preds[preds<0.5] = 0
        temp_preds[preds>=0.5] = 1
        
        n_correct = (targets == temp_preds).sum()
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


