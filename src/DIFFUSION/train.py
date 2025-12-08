import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import DentalDataset
from globals import IMAGE_SIZE, TIME_STEPS, BATCH_SIZE, DEVICE, DATA_PATH
from network import Diffusion, EMA

OUTPUT_DIR = "outputs/running"
TOTAL_STEPS_EMA = 100000
TOTAL_STEPS_CLASSIC = 50000
LR_EMA = 8e-5
LR_CLASSIC = 1e-4
SAVE_EVERY = 1000
SHOW_EVERY = 100
PRINT_EVERY = 1
RESUME_CKPT = ""  
EMA_DECAY = 0.999         

if __name__ == "__main__":

    use_ema = input("Select the modality: 1 for EMA DDPM - 0 for classical DDPM ").strip()
    if use_ema == "1":
        print("EMA DDPM choosen")
        CHECKPOINT_DIR = "DDPM_EMA"
        LR = LR_EMA
        total_steps = TOTAL_STEPS_EMA
    else:
        use_ema = 0
        print("Classic DDPM choosen")
        CHECKPOINT_DIR = "DDPM"
        LR = LR_CLASSIC
        total_steps = TOTAL_STEPS_CLASSIC

    # Dataset and dataloader initialization
    dataset = DentalDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    print(f"Dataset loaded: {DATA_PATH}")
    
    # Model instantiation
    diffusion = Diffusion(image_size=IMAGE_SIZE, timesteps=TIME_STEPS, device=DEVICE)
    optimizer = torch.optim.Adam(diffusion.model.parameters(), lr=LR)

    ema = None
    if use_ema:
        ema = EMA(diffusion.model, EMA_DECAY)

    # Create the checkpoints dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_base = os.path.join("checkpoints", CHECKPOINT_DIR, "v2")
    os.makedirs(ckpt_base, exist_ok=True)

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

        if use_ema and "ema" in ckpt:
            ema.shadow = ckpt["ema"]
            print("EMA loaded.")

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

            # Loss computing and backpropagation
            optimizer.zero_grad()
            loss = diffusion.p_losses(imgs, t)
            loss.backward()
            optimizer.step()

            # Step increment
            step += 1

            # Ema weights update
            if use_ema:
                ema.update()

            # Logging
            if step % PRINT_EVERY == 0:
                print(f"Step {step:05d} | Loss: {loss.item():.6f}")

            # Save a photo every SHOW_EVERY steps
            if step % SHOW_EVERY == 0:

                diffusion.model.eval()
                if use_ema:
                    ema.apply_shadow()
                sample = diffusion.sample(n_samples=1)
                if use_ema:
                    ema.restore()
                diffusion.model.train()

                # Sample normalization to visualize it
                sample = (sample.clamp(-1, 1) + 1) / 2

                for i in range(sample.shape[0]):
                    img = sample[i, 0].cpu().numpy()
                    save_path = os.path.join(OUTPUT_DIR, f"{step}.png")
                    plt.imsave(save_path, img, cmap="gray")
                    print(f"Saved: {save_path}")

            # Checkpoint save
            if step % SAVE_EVERY == 0:
                ckpt_path = os.path.join(
                    ckpt_base, f"{IMAGE_SIZE}_diffusion_step_{step}_{TIME_STEPS}_timesteps.pt"
                )
                ckpt = {
                    "model_state_dict": diffusion.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step
                }
                if use_ema and ema is not None:
                    ckpt["ema"] = ema.shadow
                torch.save(ckpt, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

        # Stop condition
        if step >= total_steps:
            print(f"Training complete ({total_steps} steps reached)")
            break