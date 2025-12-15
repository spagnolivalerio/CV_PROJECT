import torch
from network import Generator
import torchvision.utils as vutils
import os
from utils import unnormalize
from globals import Z_DIM, DEVICE, DATA_ROOT

OUTPUT_DIR = DATA_ROOT + "/fake"
NUM_SAMPLES = 600

WEIGHTS_PATH = "weights/G64_C64_Z100_L7.pt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

weights = torch.load(WEIGHTS_PATH, map_location="cuda")

G = Generator(device=DEVICE).to(DEVICE)
G.load_state_dict(weights["gen"])
G.eval()

for i in range(NUM_SAMPLES):
    z = torch.randn(1, Z_DIM, device=G.device)
    with torch.no_grad():
        fake = unnormalize(G(z).detach().cpu())
        fake = torch.clamp(fake, 0, 1).squeeze()
    vutils.save_image(fake, f"{OUTPUT_DIR}/synthetic_{i}.png", normalize=True)