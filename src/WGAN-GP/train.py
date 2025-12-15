from network import Generator, Critic
from globals import DEVICE, Z_DIM, DATA_PATH, C_CHANNELS, G_CHANNELS, LAMBDA
from torch.utils.data import DataLoader
from dataset import DentalDataset
from torchvision.utils import save_image
from utils import unnormalize
import torch
import os

GLR = 1e-4
CLR = 5e-5
BATCH_SIZE = 16
EPOCHS = 1000
CRITIC_STEPS = 2
SAVE_EVERY = 10
WEIGHTS_PATH = "weights"

SAMPLES_PATH = os.path.join("data", "outputs", "runtime", f"G{G_CHANNELS}_C{C_CHANNELS}_Z{Z_DIM}")

if __name__ == "__main__":
    
    os.makedirs(WEIGHTS_PATH, exist_ok=True)
    os.makedirs(SAMPLES_PATH, exist_ok=True)

    G = Generator(device=DEVICE).to(DEVICE)
    C = Critic(device=DEVICE).to(DEVICE)

    c_optimizer = torch.optim.Adam(C.parameters(), lr=CLR, betas=(0.0, 0.9)) # Taken from paper
    g_optimizer = torch.optim.Adam(G.parameters(), lr=GLR, betas=(0.0, 0.9))

    dataset = DentalDataset(DATA_PATH)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    print("Starting training...")
    for e in range(EPOCHS):
        
        for i, (img) in enumerate(dataloader):

            real = img.to(DEVICE)
            critic_loss, gp, fake_score, real_score = C.train_on_batch(G, CRITIC_STEPS, Z_DIM, real, c_optimizer)
            generator_loss = G.train_on_batch(C, real.size(0), g_optimizer)
            print(f"EPOCH {e+1}/{EPOCHS} [{i}/{len(dataloader)}] - G: {generator_loss:.2f} - C: {critic_loss:.2f} ({fake_score:.2f} - {real_score:.2f} + {LAMBDA}*{gp:.2f}) ")

        print(f"Epoch {e+1} completed")

        # Sample and save a grid of generated images every epoch
        with torch.no_grad():
            G.eval()
            z = torch.randn(4, Z_DIM, device=C.device)
            samples = G(z)
            samples = unnormalize(samples).clamp(0, 1)
            save_image(samples, os.path.join(SAMPLES_PATH, f"samples_epoch_{e+1:04d}.png"), nrow=4)
            G.train()

        if (e + 1) % SAVE_EVERY == 0:
            torch.save({
                "gen": G.state_dict()
            }, f"{WEIGHTS_PATH}/G{G_CHANNELS}_C{C_CHANNELS}_Z{Z_DIM}.pt")
            print("Saved")
            
            
