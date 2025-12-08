import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import DentalDataset
from globals import IMAGE_SIZE, TIME_STEPS, BATCH_SIZE, DEVICE, DATA_PATH
from network import Diffusion

DATA_PATH = "../../data/dentex/training_data/quadrant/xrays/"
OUTPUT_DIR = "outputs/running"
TOTAL_STEPS = 50000
LR = 1e-4
SAVE_EVERY = 1000
SHOW_EVERY = 100
PRINT_EVERY = 1
RESUME_CKPT = ""           

dataset = DentalDataset(DATA_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
print(f"Dataset loaded: {DATA_PATH}")

diffusion = Diffusion(image_size=IMAGE_SIZE, timesteps=TIME_STEPS, device=DEVICE)
optimizer = torch.optim.Adam(diffusion.model.parameters(), lr=LR)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Resume checkpoint (if needed)
start_step = 0
if RESUME_CKPT and os.path.exists(RESUME_CKPT):
    print(f"Checkpoint found: {RESUME_CKPT}")

    # Checkpoint location and loading device
    ckpt = torch.load(RESUME_CKPT, map_location=DEVICE)
    diffusion.model.load_state_dict(ckpt["model_state_dict"])
    print("Model loaded.")

    # Optimizer state loading
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print("Optimizer loaded.")

    # Taking the start step
    start_step = ckpt.get("step", 0)
    print(f"Resumed from step {start_step}")
else:
    print("No checkpoint found. Starting training from scratch.")

print("Work in progress, trust the process ;)")

# Start training
step = start_step
diffusion.model.train()

for epoch in range(999999):
    for i, imgs in enumerate(dataloader):

        imgs = imgs.to(DEVICE)
        
        # Create a t array to contain timestep for each image in the batch
        t = torch.randint(0, TIME_STEPS, (imgs.shape[0],), device=DEVICE) 
        
        # Training step
        optimizer.zero_grad()
        loss = diffusion.p_losses(imgs, t)
        loss.backward()
        optimizer.step()
        
        step += 1

        if step % PRINT_EVERY == 0:
            print(f"Step {step:05d} | Loss: {loss.item():.6f}")

        # Save a photo every SHOW_EVERY steps
        if step % SHOW_EVERY == 0:
            diffusion.model.eval()
            sample = diffusion.sample(n_samples=1)

            # Sample normalization to visualize it
            sample = (sample.clamp(-1, 1) + 1) / 2

            for i in range(sample.shape[0]):
                img = sample[i, 0].cpu().numpy()
                save_path = os.path.join(OUTPUT_DIR, f"{step}.png")
                plt.imsave(save_path, img, cmap="gray")
                print(f"Saved: {save_path}")
            
            diffusion.model.train()

        # Checkpoint save
        if step % SAVE_EVERY == 0:
            ckpt_path = f"checkpoints/DDPM/1000_timesteps/{IMAGE_SIZE}_diffusion_step_{step}_{TIME_STEPS}_timesteps.pt"
            torch.save({
                "model_state_dict": diffusion.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        if step >= TOTAL_STEPS:
            print(f"Training complete ({TOTAL_STEPS} steps reached)")
            break

    if step >= TOTAL_STEPS:
        break
