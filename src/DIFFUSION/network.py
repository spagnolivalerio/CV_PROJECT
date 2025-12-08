from diffusers import UNet2DModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from globals import IMAGE_SIZE, TIME_STEPS

class Diffusion(nn.Module):
    def __init__(self, image_size=IMAGE_SIZE, timesteps=TIME_STEPS, beta_start=1e-4, beta_end=0.02, device="cuda"):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.timesteps = timesteps

        self.model = UNet2DModel(
            sample_size=image_size,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(32, 64, 128), 
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        ).to(device)

        # Noise scheduler
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    # Forward process
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_bar = self.alpha_bars[t][:, None, None, None].sqrt() # Broadcast shape
        sqrt_one_minus_alpha_bar = (1 - self.alpha_bars[t])[:, None, None, None].sqrt()
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise, noise

    # Loss function
    def p_losses(self, x0, t):
        x_t, noise = self.q_sample(x0, t)
        noise_pred = self.model(x_t, t).sample
        loss = F.mse_loss(noise_pred, noise)
        return loss

    # Reverse process
    @torch.no_grad()
    def sample(self, n_samples):
        self.model.eval()
        x = torch.randn(n_samples, 1, self.image_size, self.image_size, device=self.device)

        for t in reversed(range(self.timesteps)):
            t_tensor = torch.tensor([t] * n_samples, device=self.device).long()
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]

            noise_pred = self.model(x, t_tensor).sample
            noise = torch.randn_like(x) if t > 0 else 0

            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
            x = coef1 * (x - coef2 * noise_pred) + torch.sqrt(beta_t) * noise

        return x
    
class EMA:
    def __init__(self, model, decay):
        """
        Initialize EMA class to manage exponential moving average of model parameters.
        
        Args:
            model (torch.nn.Module): The model for which EMA will track parameters.
            decay (float): Decay rate, typically a value close to 1, e.g., 0.999.
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Store initial parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        """
        Update shadow parameters with exponential decay.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    @torch.no_grad()
    def apply_shadow(self):
        """
        Apply shadow (EMA) parameters to model.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    @torch.no_grad()
    def restore(self):
        """
        Restore original model parameters from backup.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

