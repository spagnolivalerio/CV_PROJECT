import os
import torch
import matplotlib.pyplot as plt
from network import Diffusion, EMA
from globals import CHECKPOINTS_ROOT, DATA_ROOT
from train import EMA_DECAY
from utils import invert_normalization

N_SAMPLES = 1
VERSION = 1
CHECKPOINT_PATH = CHECKPOINTS_ROOT + "/DDPM_EMA/v1/128_diffusion_step_35000_450_timesteps.pt"
OUTPUT_DIR = DATA_ROOT + f"/outputs/samples_{VERSION}"


if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    diffusion = Diffusion()
    model = diffusion.model

    model.eval()
    ema = EMA(model, EMA_DECAY)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=model.device)

    if not ckpt["ema"]:
        raise Exception("No ema weights found")

    ema.shadow = ckpt["ema"]
    ema.apply_shadow()

    outputs = diffusion.sample_image(n_samples=N_SAMPLES)

    outputs = invert_normalization(outputs)
    outputs = torch.clamp(outputs, 0, 1).squeeze()

    img = outputs.cpu().numpy()
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()



