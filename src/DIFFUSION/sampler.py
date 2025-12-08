import os
import torch
import matplotlib.pyplot as plt
from DIFFUSION.network import Diffusion 
from utils import IMAGE_SIZE
from utils import TIME_STEPS
from DIFFUSION.train import EMA, EMA_DECAY

N_SAMPLE = 1
CYCLES = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = f"checkpoints/DDPM_EMA/v1/128_diffusion_step_35000_450_timesteps.pt"
OUTPUT_DIR = f"data/outputs/samples_{N_SAMPLE}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

diffusion = Diffusion(
    image_size=IMAGE_SIZE,
    timesteps=TIME_STEPS,
    beta_start=1e-4,
    beta_end=0.02,
    device=DEVICE
)

ema = EMA(diffusion.model, EMA_DECAY)

print(f"Load from: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
diffusion.model.load_state_dict(checkpoint["model_state_dict"])
ema.shadow = checkpoint['ema']
step = checkpoint.get("step")
print(f"Checkpoint loaded (step = {step})")

print("Generating new images...")

diffusion.model.eval()
ema.apply_shadow()

for c in range(CYCLES):

    with torch.no_grad():
        samples = diffusion.sample(n_samples=10)

    samples = (samples.clamp(-1, 1) + 1) / 2

    for i in range(samples.shape[0]):
        img = samples[i, 0].cpu().numpy()
        save_path = os.path.join(OUTPUT_DIR, f"generated_{i}_step{step}_{IMAGE_SIZE}px_CYCLE_{c}.2.png")
        plt.imsave(save_path, img, cmap="gray")
        print(f"Saved: {save_path}")

