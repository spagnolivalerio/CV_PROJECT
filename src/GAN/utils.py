import torch.autograd as autograd
import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.transforms import InterpolationMode
from globals import IMAGE_SIZE, DEVICE
import matplotlib.pyplot as plt

crop = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.NEAREST),
    transforms.CenterCrop(IMAGE_SIZE)
])

crop_and_normalize = transforms.Compose([
    crop,
    transforms.Grayscale(num_output_channels=1),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Gradient penalty of WGAN
def gradient_penalty(critic, real, fake, device=DEVICE):

    batch_size = real.shape[0]
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)

    real_fake_blend = epsilon * real + (1 - epsilon) * fake
    real_fake_blend = real_fake_blend.to(device)
    real_fake_blend.requires_grad_(True)

    mixed_scores = critic.forward(real_fake_blend)

    grad_outputs = torch.ones_like(mixed_scores, device=device)
    gradients = autograd.grad(
        outputs=mixed_scores,
        inputs=real_fake_blend,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # shape (B, C, H, W)

    gradients = gradients.view(batch_size, -1)        # (B, C*H*W)
    grad_norm = gradients.norm(2, dim=1)              # (B)

    gp = torch.mean((grad_norm - 1) ** 2)

    return gp 

def show(image, channels=1, image_size=IMAGE_SIZE, num_images=16):

    data = image.detach().cpu().view(-1, channels, image_size, image_size)
    grid = make_grid(data[:num_images], nrow=4).permute(1, 2, 0)
    plt.imshow(grid)
    plt.show()

    
