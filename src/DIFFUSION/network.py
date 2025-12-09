from diffusers import UNet2DModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from globals import IMAGE_SIZE, TIME_STEPS, DEVICE, BETA_START, BETA_END

class Diffusion(nn.Module):
    def __init__(self, image_size=IMAGE_SIZE, timesteps=TIME_STEPS, beta_start=BETA_START, beta_end=BETA_END, device=DEVICE):

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
        
        # ---------------------------------- #
        # --------- Noise scheduler -------- # 
        # We create a betas array which cointaines the variance that we have to sum
        # at the noise. betas[t] contains the component at timestep t.
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device) # Shape [T]
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0) # Shape [T] (Total timesteps)

    # ----------------------------------- #
    # --------- Forward process --------- #
    # For a single image in the batch, we have that x_t[i],the i-th image in the batch adding
    # noise at timestep t, is -->
    # x_t[i] = [sqrt(alpha_bar_t[i]) * x0[i]] + [sqrt(1 - alpha_bar_t[i]) * eps]
    # To produce a single batch, we consider alpha_bar_t as a single vector of shape B
    # (a component for each image in the batch).
    # Then, we have to expand it to compute directly the element wise product: 
    # Tensor element wise product rule: all dimensions match or someone is 1: 
    # alpha_bar[t] has shape [B, 1, 1, 1], the batch has shape [B, C, H, W].
    # For eache image i in the batch Batch[i, C, H, W], we perform scalar product by the scalar
    # component of alpha_bar[t][i, 0, 0, 0].
    def q_sample(self, x0, t): # t has shape [B]
        """
        Used in the training loop to generate the noised image.
        Explicitly assumes `t` is provided per-sample (shape [B]): each image in the batch
        can use a different timestep, so we index alpha_bar with a vector and broadcast it
        (the operation of align the dimensions among tensors)
        """
        # Noise creation
        eps = torch.randn_like(x0)

        # alpha_bars[t], if 't' is a vector, returns all values indexing with values of t
        sqrt_alphas_bar = self.alphas_bar[t][:, None, None, None].sqrt()
        sqrt_one_minus_alphas_bar = (1 - self.alphas_bar[t])[:, None, None, None].sqrt()
        return sqrt_alphas_bar * x0 + sqrt_one_minus_alphas_bar * eps, eps

    # Loss function
    def loss(self, x0, t): # Shape x0: [B, C, H, W], t: [B]

        # Produce the noised image at timestep 't' through the diffusion process 
        x_t, eps = self.q_sample(x0, t)
        noise_pred = self.model(x_t, t).sample # sample is a function of UNet2DModel istance
        loss = F.mse_loss(noise_pred, eps)

        return loss

    # Reverse process
    @torch.no_grad()
    def sample_image(self, n_samples):

        """
        Used in the sampler function to generate an arbitrary number of samples.
        """
        self.model.eval()

        # Generate the normal noise N(0,1) to start the reverse process of shape [B, C, H, W]
        x = torch.randn(n_samples, 1, self.image_size, self.image_size, device=self.device)

        for t in reversed(range(self.timesteps)):

            # Create an array [t, t, ..., t] of shape n_samples (B)
            # Here we assume a single timestep shared across the batch at this loop iteration
            # (scalar t), then expand it to per-sample vector because the UNet expects a t per sample.
            t_vector = torch.ones(n_samples, device=self.device, dtype=torch.long) * t
            betas_t = self.betas[t]
            alphas_t = self.alphas[t]
            alphas_bar_t = self.alphas_bar[t]

            noise_pred = self.model(x, t_vector).sample
            
            # Generate a gaussian normal noise
            noise = torch.randn_like(x) if t > 0 else 0

            # Following the reverse process from literature
            coef1 = 1 / torch.sqrt(alphas_t)
            coef2 = (1 - alphas_t) / torch.sqrt(1 - alphas_bar_t)
            x = coef1 * (x - coef2 * noise_pred) + torch.sqrt(betas_t) * noise

        return x

# https://medium.com/@heyamit10/exponential-moving-average-ema-in-pytorch-eb8b6f1718eb 
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
