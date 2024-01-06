import os 
from plyfile import PlyData, PlyElement
import numpy as np
import matplotlib.pylab as plt
from dataloader import ModelNet1
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import set_seed_globally
import torch
from Model import PointNet_Encoder,Decoder,VAE
# from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch.nn.functional as F
from champfer_loss import ChamferLoss
import trimesh

device="cpu"
batch_size = 1
learning_rate=1e-5
std_assumed = torch.tensor(0.2)
std_assumed = std_assumed.to(device)

train_Dataset = ModelNet1(train_folder=False, npoints=1024)
dataloader = DataLoader(train_Dataset, batch_size=batch_size,shuffle=True)

encoder=PointNet_Encoder().to(device)
decoder=Decoder().to(device)
model = VAE(encoder,decoder,latent_dim=64).to(device)

checkpoint = torch.load("VAE_LOGS/Model/best_model.pth", map_location='cpu')
model.load_state_dict(checkpoint["model_state_dict"])
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# criterion = ChamferLoss()

def recon_plot(rx,x):
    x, rx = x.detach().numpy()[0], rx.detach().numpy()[0]
    # print(x.shape,rx.shape)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection': '3d'})
    
    for ax, data, title in zip(axes, [x, rx], ['Ground Truth', 'Reconstructed']):
        ax.scatter(*data.T, s=5)
        ax.set_title(title)
    
    plt.tight_layout()
    plt.show()

def recon_plot2(x1,x2,z):
    x1= x1.detach().numpy()[0]
    x2 = x2.detach().numpy()[0]
    z=z.detach().numpy()[0]
    # print(x1.shape,z.shape)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': '3d'})
    plot_data = [x1, x2,z] 
    titles = ['Ground Truth', 'Reconstructed', 'interpolated']
    for ax, data, title in zip(axes, plot_data, titles):
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=3) 
        ax.set_title(title)
    
    plt.tight_layout()
    plt.show()

def generate_samples():
    for i,data in enumerate(dataloader,0):
        x = data.to(device).float()
        x_recon, mu, logvar = model(x.transpose(2,1))
        # for j in range(10):
        #     latent= model.reparameterize(mu,logvar)
        #     out=model.decoder(latent)
        #     recon_plot(out.permute(0,2,1),x)
        recon_plot(x_recon.permute(0,2,1),x)
        if i==4:
            break

def random_sample():
    x=np.random.randn(1,1024,3)
    x=torch.from_numpy(x)
    x=x.to(device).float()
    x_recon, mu, logvar = model(x.transpose(2,1))
    recon_plot(x_recon.permute(0,2,1),x)

def random_sample2():
    x=np.random.randn(1,64)
    x=torch.from_numpy(x)
    x=x.to(device).float()
    z=model.decoder(x)
    recon_plot(z,z)

def interpolate(autoencoder, x_1, x_2, n=12):
    a,b = autoencoder.encoder(x_1)
    c,d= autoencoder.encoder(x_2)
    z1=model.reparameterize(a,b)
    z2=model.reparameterize(c,d)
    z = torch.stack([z1 + (z2 - z1)*t for t in np.linspace(0, 1, n)])
    q = autoencoder.decoder(z)
    for i in range(n):
        v=q[i].unsqueeze(0)
        recon_plot2(x_1.transpose(2,1),x_2.transpose(2,1),v.transpose(2,1))

def just_plot(a,b):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection': '3d'})
    
    for ax, data, title in zip(axes, [a,b], ['Ground Truth', 'Reconstructed']):
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=5) 
        ax.set_title(title)
    
    plt.tight_layout()
    plt.show()


if __name__=="__main__":
    generate_samples()
    # random_sample()
    # a=[]
    # for i,data in enumerate(dataloader,0):
    #     x = data.to(device).float()
    #     a.append(x.transpose(2,1))
    # interpolate(model,a[0],a[1])
    # a=train_Dataset[0]
    # b=train_Dataset[1]
    # just_plot(a,b)
