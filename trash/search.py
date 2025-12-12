import itertools
import json
import os
import select
import sys
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

"""import globals
from dataset import DentalDataset
from network import Generator, Critic
from utils import unnormalize
import network as net
"""
LR = 1e-4
BATCH_SIZE = 16
EPOCHS = 1000
CRITIC_STEPS = 4
SAVE_EVERY = 50

HYPERSPACE = {
    "G_CHANNELS": [512, 1024, 2048],
    "Z_DIM": [256],
    "C_CHANNELS": [32, 64, 128],
    "LAMBDA": [10],
}

BASE_OUT = os.path.join("outputs", "search")
CONF_LOG = os.path.join(BASE_OUT, "conf.json")
SKIP_COMMANDS = {"s", "skip", "next"}


def _should_skip_current_config():
    """Non-blocking check for user input requesting a skip."""
    if not sys.stdin.isatty():
        return False
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if ready:
        user_input = sys.stdin.readline().strip().lower()
        return user_input in SKIP_COMMANDS
    return False


def run_config(g_channels, z_dim, c_channels, gp_lambda):
    config_id = f"G{g_channels}_C{c_channels}_Z{z_dim}_L{gp_lambda}"
    exp_dir = os.path.join(BASE_OUT, config_id)
    weights_dir = os.path.join(exp_dir, "weights")
    samples_dir = exp_dir  # folder for images named with the config id
    log_path = os.path.join(exp_dir, "logs.txt")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    # Persist the configuration for traceability in a shared JSON list
    config_entry = {
        "id": config_id,
        "conf": {
            "G_CHANNELS": g_channels,
            "Z_DIM": z_dim,
            "C_CHANNELS": c_channels,
            "LAMBDA": gp_lambda,
            "LR": LR,
            "BATCH_SIZE": BATCH_SIZE,
            "EPOCHS": EPOCHS,
            "CRITIC_STEPS": CRITIC_STEPS,
        },
    }
    if os.path.exists(CONF_LOG):
        with open(CONF_LOG, "r") as f:
            try:
                conf_data = json.load(f)
            except json.JSONDecodeError:
                conf_data = []
    else:
        conf_data = []
    conf_data = [entry for entry in conf_data if entry.get("id") != config_id]
    conf_data.append(config_entry)
    with open(CONF_LOG, "w") as f:
        json.dump(conf_data, f, indent=2)

    # Set lambda for gradient penalty inside the imported modules
    globals.LAMBDA = gp_lambda
    net.LAMBDA = gp_lambda

    device = globals.DEVICE
    dataset = DentalDataset(globals.DATA_PATH)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    G = Generator(z_dim=z_dim, start_channels=g_channels, out_channels=globals.OUT_CHANNELS, device=device).to(device)
    C = Critic(start_channels=c_channels, device=device).to(device)

    c_optimizer = torch.optim.Adam(C.parameters(), lr=LR, betas=(0.0, 0.9))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.0, 0.9))

    logs = []
    print(f"Starting config {config_id}")
    for e in range(EPOCHS):
        epoch_g_loss = 0.0
        epoch_c_loss = 0.0
        epoch_gp = 0.0
        batches = 0
        for i, (img) in enumerate(dataloader):
            real = img.to(device)
            critic_loss, gp, fake_score, real_score = C.train_on_batch(G, CRITIC_STEPS, z_dim, real, c_optimizer)
            generator_loss = G.train_on_batch(C, real.size(0), g_optimizer)
            epoch_g_loss += generator_loss
            epoch_c_loss += critic_loss
            epoch_gp += gp
            batches += 1
            print(f"[{config_id}] EPOCH {e+1}/{EPOCHS} [{i}/{len(dataloader)}] - G: {generator_loss:.2f} - C: {critic_loss:.2f} ({fake_score:.2f} - {real_score:.2f} + 10*{gp:.2f})")

        # Aggregate per-epoch stats
        logs.append(
            {
                "epoch": e + 1,
                "gen_loss": epoch_g_loss / max(batches, 1),
                "critic_loss": epoch_c_loss / max(batches, 1),
                "gp": epoch_gp / max(batches, 1),
            }
        )

        with torch.no_grad():
            G.eval()
            z = torch.randn(16, z_dim, device=device)
            samples = G(z)
            samples = unnormalize(samples).clamp(0, 1)
            save_image(samples, os.path.join(samples_dir, f"samples_epoch_{e+1:04d}.png"), nrow=4)
            G.train()

        if (e + 1) % 10 == 0 or e + 1 == EPOCHS:
            with open(log_path, "w") as f:
                for entry in logs:
                    f.write(
                        f"epoch={entry['epoch']} "
                        f"gen_loss={entry['gen_loss']:.4f} "
                        f"critic_loss={entry['critic_loss']:.4f} "
                        f"gp={entry['gp']:.4f}\n"
                    )

        if (e + 1) % SAVE_EVERY == 0 or e + 1 == EPOCHS:
            torch.save({"gen": G.state_dict(), "critic": C.state_dict()}, os.path.join(weights_dir, f"weights_epoch_{e+1:04d}.pt"))
            print(f"[{config_id}] Saved weights at epoch {e+1}")

        if _should_skip_current_config():
            print(f"[{config_id}] Skip requested; moving to the next configuration.")
            break

    # Free a bit of memory between runs
    del G, C
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


if __name__ == "__main__":
    os.makedirs(BASE_OUT, exist_ok=True)
    for g_channels, z_dim, c_channels, gp_lambda in itertools.product(
        HYPERSPACE["G_CHANNELS"],
        HYPERSPACE["Z_DIM"],
        HYPERSPACE["C_CHANNELS"],
        HYPERSPACE["LAMBDA"],
    ):
        run_config(g_channels, z_dim, c_channels, gp_lambda)
