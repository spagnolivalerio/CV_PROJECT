import torch
from models.gan import Generator
import torchvision.utils as vutils
import os

z_dim = 128

device = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "outputs"
WEIGHTS_PATH = ""

os.makedirs(OUTPUT_DIR, exist_ok=True)

gen = Generator().to(device)
gen.load_state_dict(torch.load(WEIGHTS_PATH))
gen.eval()

num_images = 50

for i in range(num_images):
    z = torch.randn(1, z_dim, 1, 1, device=device)
    with torch.no_grad():
        fake = gen(z).detach().cpu()
    vutils.save_image(fake, f"{OUTPUT_DIR}/synthetic_{i}.png", normalize=True)