"""
Esegue l'inferenza della rete di segmentazione sulle immagini generate
dal diffusion model e visualizza per ogni campione:
- immagine originale
- maschera predetta
- overlay immagine + maschera
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Optional


ROOT = Path(__file__).resolve().parent
SEGMENTATION_DIR = ROOT / "SEGMENTATION"

# Garantisce che gli import facciano riferimento ai file della cartella SEGMENTATION
sys.path.append(str(SEGMENTATION_DIR))

from utils import (  # type: ignore  # noqa: E402
    classes_to_palette,
    color_mask,
    crop_and_normalize,
    make_overlay,
    tensor_to_image,
)
import globals as seg_globals  # type: ignore  # noqa: E402


DEFAULT_FAKE_DIR = ROOT / "DIFFUSION" / "data" / "fake"
DEFAULT_WEIGHTS = SEGMENTATION_DIR / "weights" / "unet_legacy.pt"


class FakeImagesDataset(Dataset):
    """Dataset minimale per immagini di test senza maschere di GT."""

    def __init__(self, imgs_dir: Path):
        self.img_files = sorted(
            [p for p in imgs_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        )
        if not self.img_files:
            raise FileNotFoundError(f"Nessuna immagine trovata in {imgs_dir}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert("L")
        tensor = crop_and_normalize(img)
        return tensor, img_path.name


def plot_inference(original_img, colored_mask, overlay, title: Optional[str] = None):
    plt.figure(figsize=(12, 4))

    if title:
        plt.suptitle(title)

    plt.subplot(1, 3, 1)
    plt.imshow(original_img, cmap="gray")
    plt.title("Immagine")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(colored_mask)
    plt.title("Mask predetta")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def save_outputs(original_img, colored_mask, overlay, save_dir: Path, stem: str):
    save_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(original_img).save(save_dir / f"{stem}_original.png")
    Image.fromarray(colored_mask).save(save_dir / f"{stem}_mask.png")

    overlay_img = (overlay * 255).clip(0, 255).astype("uint8")
    Image.fromarray(overlay_img).save(save_dir / f"{stem}_overlay.png")


def load_model(weights_path: Path, device: torch.device):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=seg_globals.NUM_CLASSES,
    ).to(device)

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def run_inference(args):
    device = torch.device(seg_globals.DEVICE if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("CUDA non disponibile: uso CPU.")

    model = load_model(args.weights, device)
    palette = classes_to_palette()

    dataset = FakeImagesDataset(args.images_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for img_tensor, name in dataloader:
            img_tensor = img_tensor.to(device)
            preds = model(img_tensor)
            pred_mask = torch.argmax(preds, dim=1).squeeze(0).cpu().numpy()

            original_img = tensor_to_image(img_tensor)
            colored_mask = color_mask(pred_mask, palette)
            overlay = make_overlay(original_img, colored_mask)

            plot_inference(original_img, colored_mask, overlay, title=name[0])
            if args.save_dir:
                stem = Path(name[0]).stem
                save_outputs(original_img, colored_mask, overlay, args.save_dir, stem)


def parse_args():
    parser = argparse.ArgumentParser(description="Test della rete di segmentazione su immagini fake.")
    parser.add_argument(
        "--images_dir",
        type=Path,
        default=DEFAULT_FAKE_DIR,
        help="Cartella con le immagini generate dal diffusion model.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Percorso ai pesi della rete di segmentazione.",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=None,
        help="Se specificata, salva immagine, mask e overlay in questa cartella.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
