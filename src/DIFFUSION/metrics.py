import os
from glob import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
from cleanfid import fid
from torchmetrics.image.inception import InceptionScore
from utils import crop_and_resize

# Inception V3 requires 299x299, RGB
inseption_score_transformation = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.PILToTensor() 
])

# https://github.com/GaParmar/clean-fid
def compute_clean_FID(REAL_DIR, FAKE_DIR, convert_images=False):

    # If the REAL DIR has native images
    if convert_images:

        files = sorted(glob(os.path.join(REAL_DIR, "*")))

        for file in files:
            # Open the image in grayscale
            img = Image.open(file).convert("L")

            # Convert the image in [0, 1] domain (FID requires it) and resize it
            img = crop_and_resize(img)
            img = torch.clamp(img, 0, 1).squeeze(0)

            # Trasform to PIL format to save it
            img = transforms.ToPILImage()(img)
            img.save(file)

    # Compute the FID
    fid_value = fid.compute_fid(REAL_DIR, FAKE_DIR, mode="clean")
    print(f"FID VALUE: {fid_value:.4f}")

# https://lightning.ai/docs/torchmetrics/stable/image/inception_score.html
def compute_IS(REAL_DIR, FAKE_DIR):

    inception_real = InceptionScore()
    inception_fake = InceptionScore()
    inception_noise = InceptionScore()

    real_images = []
    fake_images = []

    real_files = sorted(glob(os.path.join(REAL_DIR, "*")))
    fake_files = sorted(glob(os.path.join(FAKE_DIR, "*")))

    for f in real_files[:50]:
        # PIL format
        real_images.append(inseption_score_transformation(Image.open(f).convert("RGB")))

    for f in fake_files[:50]:
        # PIL format
        fake_images.append(inseption_score_transformation(Image.open(f).convert("RGB")))
    
    # Compute the batch as the input of IS
    real_batch = torch.stack(real_images, dim=0) # Shape [B, C, H, W]
    fake_batch = torch.stack(fake_images, dim=0)
    noise_batch = torch.randint(0, 255, (50, 3, 299, 299), dtype=torch.uint8)
    
    inception_real.update(real_batch)
    inception_fake.update(fake_batch)
    inception_noise.update(noise_batch)

    real_score, _  = inception_real.compute()
    fake_score, _  = inception_fake.compute()
    noise_score, _ = inception_noise.compute()

    print(
            f"   IS REAL VALUE: {float(real_score):.4f}"
            f" | IS FAKE VALUE: {float(fake_score):.4f}" 
            f" | IS NOISE VALUE: {float(noise_score):.4f}"
        )
        
