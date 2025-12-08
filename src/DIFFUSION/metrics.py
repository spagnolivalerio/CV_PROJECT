import os
from glob import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
from cleanfid import fid
from utils import crop_and_resize

def compute_clean_FID(REAL_DIR, FAKE_DIR, convert_images=False):

    # If the REAL DIR has native images
    if convert_images:

        files = sorted(glob(os.path.join(REAL_DIR, "*")))

        for file in files:
            # Open the image in grayscale
            img = Image.open(file).convert("L")

            # Convert the image in [0, 1] domain and resize it
            img = crop_and_resize(img)
            img = torch.clamp(img, 0, 1).squeeze(0)

            # Trasform to PIL format to save it
            img = transforms.ToPILImage()(img)
            img.save(file)

    # Compute the FID
    fid_value = fid.compute_fid(REAL_DIR, FAKE_DIR, mode="clean")
    print(f"FID VALUE: {fid_value:.4f}")
    