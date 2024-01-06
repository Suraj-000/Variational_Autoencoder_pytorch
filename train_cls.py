from datetime import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import ModelNet1
from champfer_loss import ChamferLoss
from Model import PointNet_Encoder,Decoder,VAE
from utils import *

def main():
    results_dir = "/workspace/VAE/VAE_LOGS"
    results = save_results(results_dir)
    logger = LOGGING(results_dir)
    device = set_seed_globally(gpu=0)

    ### Basic initialization of metrics
    best_loss=100000.0
    best_epoch=0
    best_kld_loss=best_loss
    best_recon_loss=best_loss
    epochs=1000
    batch_size = 32
    learning_rate=1e-5

    std_assumed = torch.tensor(0.2)
    std_assumed = std_assumed.to(device)
    #### DATALOADER
    train_Dataset = ModelNet1(train_folder=True, npoints=1024)
    dataloader = DataLoader(train_Dataset, batch_size=batch_size, shuffle=True, num_workers=10)

    ### Model
    encoder=PointNet_Encoder().to(device)
    decoder=Decoder().to(device)
    model = VAE(encoder,decoder,latent_dim=64).to(device)

    ### Optimizers
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    ### Loss Function
    criterion = ChamferLoss()
    
    ### Training
    logger.LOG(f"Start Training {datetime.now()}")
    for epoch in range(epochs):
        logger.LOG('Epoch (%d/%d):' % ( epoch + 1, epochs))
        model = model.train()
        for i,data in tqdm(enumerate(dataloader,0),total=len(dataloader), smoothing=0.9):
            x= data.to(device).float()
            optimizer.zero_grad()
            x_recon, mu, logvar = model(x.transpose(2,1))
            recon_loss =  criterion(x_recon.permute(0, 2, 1) + 0.5,x.permute(0, 1, 2) + 0.5)
            kl_loss = -0.5 * torch.mean(1 - 2.0 * torch.log(std_assumed) + logvar -(mu.pow(2) + logvar.exp()) / torch.pow(std_assumed, 2))
            loss = recon_loss + kl_loss
            loss.backward()
            optimizer.step()

        instance_loss=loss.item()
        if(instance_loss<=best_loss):
            best_loss=instance_loss
            best_epoch=epoch+1
        if(kl_loss<=best_kld_loss):
            best_kld_loss=kl_loss
        if(recon_loss<=best_recon_loss):
            best_recon_loss=recon_loss

        logger.LOG(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, recon_loss: {recon_loss:.4f}, KLD_loss:{kl_loss:.4f}')
        logger.LOG(f'Best Epoch {best_epoch}, Best Loss: {best_loss:.4f},Best recon_loss: {best_recon_loss:.4f}, Best KLD_loss:{best_kld_loss:.4f}')

        if(instance_loss <= best_loss):
            best_loss=instance_loss
            logger.LOG('Save model...')
            savepath = results_dir + '/Model/best_model.pth'
            logger.LOG('Saving at %s' % savepath)
            
            state = {
                'epoch': epoch,
                'Loss':best_loss,
                'Recon_Loss':recon_loss,
                'KLD_Loss':kl_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
    logger.LOG(f'End of training...{datetime.now()}')

if __name__=="__main__":
    main()
