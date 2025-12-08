import torch
import torch.optim as optim
from dataset import DentalDataset
from torch.utils.data import DataLoader
from models.gan import Generator, Critic
import torchvision.utils as vutils


DATA_ROOT= "../../data/train/xrays"
BATCH_SIZE = 16
workers = 2
z_dim = 256
gen_lr = 1e-4
crit_lr = 1e-4
lambda_gp = 10
critic_iterations = 3
EPOCHS = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialization of WGAN-GP's networks
gen = Generator().to(device)
critic = Critic().to(device)
opt_gen = optim.Adam(gen.parameters(), lr=gen_lr, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=crit_lr, betas=(0.0, 0.9))

# Dataset and dataloader initialization
dataset = DentalDataset(DATA_ROOT)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers)


for epoch in range(EPOCHS):

    for batch_idx, real in enumerate(dataloader):
        real = real.to(device)
        cur_batch_size = real.size(0) # current batch size for non batch_size multiples

        loss_critic = critic.critic_step(critic_iterations, z_dim, cur_batch_size, real, gen, opt_critic, device)

        # Generator training
        z = torch.randn(cur_batch_size, z_dim, 1, 1, device=device)
        fake = gen(z)
        output = critic(fake)
        loss_gen = -output.mean()

        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 10 == 0 or batch_idx == len(dataloader) - 1:
            print(
                f"Epoch [{epoch}/{EPOCHS}] Batch {batch_idx}/{len(dataloader) - 1} "
                f"Loss D: {loss_critic:.4f}, Loss G: {loss_gen.item():.4f}"
            )

    with torch.no_grad():
        fake_images = gen(torch.randn(5, z_dim, 1, 1, device=device)).detach().cpu()
        vutils.save_image(fake_images, f"synimages/fake_{epoch}.png", normalize=True, nrow=5)

    print(f"\nEpoch {epoch} completed.\n")

torch.save(gen.state_dict(), "weights/generator3_(1000epochs).pt")

