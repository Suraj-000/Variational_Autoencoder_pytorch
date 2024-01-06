import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary

class PointNet_Encoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(PointNet_Encoder, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, stride=1)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, stride=1)
        self.conv5 = nn.Conv1d(128,1024, kernel_size= 1, stride=1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        
        self.fc_mean = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

    def forward(self, pc):
        B,D,N = pc.shape
        feat = F.relu(self.bn1( self.conv1( pc ) ))
        feat = F.relu(self.bn2( self.conv2( feat ) ))
        feat = F.relu(self.bn3( self.conv3( feat ) ))
        feat = F.relu(self.bn4( self.conv4( feat ) ))
        feat = F.relu(self.bn5( self.conv5( feat ) ))

        x,_ = feat.max(dim=2, keepdim=False)
        mean=self.fc_mean(x)
        logvar=self.fc_logvar(x)
        return mean,logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=64,output_dim=1024):
    # def __init__(self, latent_dim=64,output_dim=2048):
        super(Decoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Linear(1024,3*output_dim),
        )

    def forward(self, x):
        x=self.layers(x)
        x = x.view(-1, 3, 1024)
        # x = x.view(-1, 3, 2048)
        return x
    
class VAE(nn.Module):
    def __init__(self,encoder,decoder,latent_dim=64):
        super(VAE,self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.latent_dim=latent_dim

    def reparameterize(self, mean, logvar):
        std = logvar.exp().sqrt()
        # eps = torch.randn(std.size()).cuda()
        eps = torch.randn(std.size()).to("cpu")
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)

        z = self.reparameterize(mean, logvar)

        x_recon = self.decoder(z)
        return x_recon, mean, logvar


