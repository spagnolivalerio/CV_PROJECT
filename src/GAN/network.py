import torch
from torch import nn

from globals import Z_DIM, OUT_CHANNELS, G_CHANNELS, C_CHANNELS, DEVICE

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
            gBlock(start_channels // 16, start_channels // 32),  # 64->128

            nn.ConvTranspose2d(start_channels // 32, out_channels, 3, 1, 1, bias=False), 
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
      
def cBlock(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
        nn.InstanceNorm2d(out_ch),
        nn.LeakyReLU(inplace=True)
    )
        
class Critic(nn.Module):
    def __init__(self, start_channels = C_CHANNELS):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, start_channels, 4, 2, 1, bias=False),  #128->64
            nn.LeakyReLU(inplace=True),
            cBlock(start_channels, start_channels*2),           #64->32
            cBlock(start_channels*2, start_channels*4),         #32->16
            cBlock(start_channels*4, start_channels*8),         #16->8
            cBlock(start_channels*8, start_channels*16),        #8->4
            cBlock(start_channels*16, start_channels*32),       #4->2
            nn.Conv2d(start_channels*32, 1, 4, 2, 1)            #2->1
        )
    
    def forward(self, img):
        out = self.model(img) # Shape [B, 1, 1, 1]
        return out.view(out.size(0)) # Shape [B]


