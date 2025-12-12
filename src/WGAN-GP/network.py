import torch
from torch import nn
from globals import Z_DIM, OUT_CHANNELS, G_CHANNELS, C_CHANNELS, DEVICE, LAMBDA
from utils import gradient_penalty

def gBlock(in_ch, out_ch, norm_layer=nn.BatchNorm2d):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False), 
        norm_layer(out_ch), 
        nn.ReLU(inplace=True) # Do not create a new tensor (saving memory)
    )

class Generator(nn.Module):

    def __init__(self, z_dim=Z_DIM, start_channels=G_CHANNELS, out_channels=OUT_CHANNELS, device=DEVICE):
        super().__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, start_channels, 4, 1, 0, bias=False),  # 1->4
            nn.BatchNorm2d(start_channels),
            nn.ReLU(inplace=True),

            gBlock(start_channels,       start_channels // 2),   # 4->8
            gBlock(start_channels // 2,  start_channels // 4),   # 8->16
            gBlock(start_channels // 4,  start_channels // 8),   # 16->32
            gBlock(start_channels // 8,  start_channels // 16),  # 32->64

            nn.ConvTranspose2d(start_channels // 16, out_channels, 4, 2, 1, bias=False), #64->128
            nn.Tanh()
        )

        self.z_dim = z_dim
        self.device = device
    
    def forward(self, noise):
        """
        Docstring for forward
        
        :param self: Generator module
        :param noise: noise with shape [B, z_dim]
        """
        B, C = noise.shape
        noise = noise.view(B, C, 1, 1)
        return self.model(noise)

    def sample(self, n_samples):

        zetas = torch.randn(n_samples, self.z_dim, device=self.device)
        return self(zetas)
    
    def train_on_batch(self, C, batch_size, generator_optimizer):

        device = next(self.parameters()).device
        z = torch.randn(batch_size, self.z_dim, device=device)
        fake = self(z)
        critic_score = C(fake)
        generator_loss = - critic_score.mean()

        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()

        return generator_loss.item()
      
# -------------------------------------------------------------------- #

def cBlock(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True)
    )
        
class Critic(nn.Module):
    def __init__(self, start_channels = C_CHANNELS, device = DEVICE):
        super().__init__()

        self.model = nn.Sequential(
            cBlock(1, start_channels), 
            cBlock(start_channels, start_channels*2),                                #128->256
            cBlock(start_channels*2, start_channels*4),                              #256->512
            cBlock(start_channels*4, start_channels*8),                              #512->1024
            cBlock(start_channels*8, start_channels*16),                             #1024->2048
            nn.Conv2d(start_channels*16, 1, 4, 1, 0)                                 #2048->1
        )

        self.device = device
    
    def forward(self, img):
        out = self.model(img) # Shape [B, 1, 1, 1]
        return out.view(out.size(0)) # Shape [B]
    
    def wasserstein_component(self, fake, real):

        fake_score = self(fake.detach()).mean()
        real_score = self(real).mean()
        
        return fake_score - real_score, fake_score, real_score

    def train_on_batch(self, G, critic_steps, z_dim, real, critic_optimizer):
        """
        Docstring for train_on_batch
        
        :param self: the Critic module
        :param G: the generator to generate fake data
        :param critic_steps: number of training steps of teh critic for each generator training step
        :param z_dim: latent space dimension
        :param real: real image from the dataset
        :param critic_optimizer: the critic optimizer, to perform weights upgrade
        """
        total_loss = 0.0
        total_gp = 0.0
        real_batch = real.size(0)

        for _ in range(critic_steps):
            
            z = torch.randn(real_batch, z_dim, device=self.device) # [B, z_dim]
            fake_out = G(z) 

            wass_comp, fake_score, real_score = self.wasserstein_component(fake_out, real)
            gp = gradient_penalty(self, real, fake_out.detach())
            loss = wass_comp + LAMBDA * gp
            critic_optimizer.zero_grad()
            loss.backward()
            critic_optimizer.step()

            total_loss += loss.item()
            total_gp += gp.item()
        
        return total_loss / critic_steps, total_gp / critic_steps, fake_score.item(), real_score.item()

