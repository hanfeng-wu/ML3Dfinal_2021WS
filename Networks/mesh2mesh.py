from numpy.lib.arraypad import pad
import torch
import torch.nn as nn


class ThreeDEPN(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_features = 80

        # TODO: 4 Encoder layers
        self.encoder_1 = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=self.num_features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv3d(in_channels=self.num_features, out_channels=2*self.num_features, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(2*self.num_features),
            nn.LeakyReLU(negative_slope=0.2),
         )
        self.encoder_3 = nn.Sequential(
            nn.Conv3d(in_channels=2*self.num_features, out_channels=4*self.num_features, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(4*self.num_features),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.encoder_4 = nn.Sequential(
            nn.Conv3d(in_channels=4*self.num_features, out_channels=8*self.num_features, kernel_size=4, stride=1),
            nn.BatchNorm3d(8*self.num_features),
            nn.LeakyReLU(negative_slope=0.2),
        )

        # TODO: 2 Bottleneck layers

        self.bottleneck = nn.Sequential(
            nn.Linear(640,640),
            nn.ReLU(),
            nn.Linear(640,640),
            nn.ReLU(),
        )

        # TODO: 4 Decoder layers

        self.decoder_1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=2*8*self.num_features, out_channels=4*self.num_features, kernel_size=4, stride=1),
            nn.BatchNorm3d(4*self.num_features),
            nn.ReLU()
        )
        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=2*4*self.num_features, out_channels=2*self.num_features, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(2*self.num_features),
            nn.ReLU()
        )
        self.decoder_3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=2*2*self.num_features, out_channels=self.num_features, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(self.num_features),
            nn.ReLU()
        )
        self.decoder_4 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=2*1*self.num_features, out_channels=1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        b = x.shape[0]
        # Encode
        # TODO: Pass x though encoder while keeping the intermediate outputs for the skip connections
        # Reshape and apply bottleneck layers
        x_e1 = self.encoder_1(x)
        x_e2 = self.encoder_2(x_e1)
        x_e3 = self.encoder_3(x_e2)
        x_e4 = self.encoder_4(x_e3)
        ####################
        x = x_e4.view(b, -1)
        x = self.bottleneck(x)
        x = x.view(x.shape[0], x.shape[1], 1, 1, 1)

        # Decode
        # TODO: Pass x through the decoder, applying the skip connections in the process
        x_d1 = self.decoder_1(torch.cat([x,x_e4],dim=1))
        x_d2 = self.decoder_2(torch.cat([x_d1,x_e3],dim=1))
        x_d3 = self.decoder_3(torch.cat([x_d2,x_e2],dim=1))
        x = self.decoder_4(torch.cat([x_d3,x_e1],dim=1))

        
        x = torch.squeeze(x, dim=1)
        # TODO: Log scaling
        x = torch.log(torch.abs(x)+1)

        return x