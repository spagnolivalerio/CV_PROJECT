import torch
import torch.nn as nn
import sys, os
from torch.nn.utils import spectral_norm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import gradient_penalty

z_dim = 256
ngf = 256    # GENERATOR filters
ndf = 32     # CRITIC filters
lambda_gp = 20

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            # z -> (ngf*8, 4, 4)
            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # 4 -> 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # 8 -> 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # 16 -> 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # 32 -> 64
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),

            # 64 -> 128 (final)
            nn.ConvTranspose2d(ngf // 2, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z)


class Critic(nn.Module):
    def __init__(self, nc=1, ndf=64):
        super().__init__()

        self.main = nn.Sequential(
            # 128 -> 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 64 -> 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 32 -> 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 16 -> 8
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            # 8 -> 4
            spectral_norm(nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            # 4 -> 1
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.main(x).view(-1)

    def critic_step(self, iterations, z_dim, batch_size, real, gen, opt_critic, device):
        total_loss = 0.0

        for _ in range(iterations):
            # fake
            z = torch.randn(batch_size, z_dim, 1, 1, device=device)
            fake = gen(z)

            # critic
            score_real = self.forward(real)
            score_fake = self.forward(fake.detach())

            # WGAN-GP loss
            gp = gradient_penalty(self, real, fake, device=device)
            loss = (score_fake.mean() - score_real.mean()) + lambda_gp * gp

            opt_critic.zero_grad()
            loss.backward()
            opt_critic.step()

            total_loss += loss.item()

        return total_loss / iterations