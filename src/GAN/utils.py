import torch.autograd as autograd
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw
from torchvision.transforms import InterpolationMode

IMAGE_SIZE = 128

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

vflip = transforms.functional.vflip

rotate_right = transforms.RandomRotation(degrees=(45, 45))

rotate_left = transforms.RandomRotation(degrees=(-45, -45))

# Gradient penalty of PANO-WGAN
def gradient_penalty(critic, real, fake, device="cuda"):

    batch_size, C, H, W = real.shape
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device, requires_grad=True)

    real_fake_blend = epsilon * real + (1 - epsilon) * fake
    real_fake_blend = real_fake_blend.to(device)

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
    grad_norm = gradients.norm(2, dim=1)              # (B,)

    gp = torch.mean((grad_norm - 1) ** 2)

    return gp 

# Compute masks starting from vertexes of polygons
def compute_masks(img, anns):

    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    for ann in anns:
        for seg in ann["segmentation"]:
            polygon = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
            draw.polygon(polygon, outline=255, fill=255)
        
    mask = np.array(mask)

    return mask


    
