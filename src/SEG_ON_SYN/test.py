import os
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
crop = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(128, interpolation=InterpolationMode.NEAREST),
    transforms.CenterCrop(128)
])

# Import relativi alla struttura del tuo progetto
crop_and_normalize = transforms.Compose([
    crop,
    transforms.Grayscale(num_output_channels=1),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

DEVICE = "cuda"


def load_model(weights_path):
    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        in_channels=1,
        classes=33,
    )
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


def predict_mask(model, img_path):
    img = Image.open(img_path).convert("L")

    # Preprocessing IDENTICO al training
    x = crop_and_normalize(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(x)  # (1, C, H, W)
        pred = torch.argmax(pred, dim=1)  # (1, H, W)

    mask = pred.squeeze(0).cpu().numpy().astype(np.uint8)
    return mask


def process_directory(input_dir, output_dir, model):
    os.makedirs(output_dir, exist_ok=True)

    images = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    print(f"Found {len(images)} images to segment.\n")

    for i, fname in enumerate(images, 1):
        path = os.path.join(input_dir, fname)

        mask = predict_mask(model, path)

        outname = os.path.splitext(fname)[0] + "_mask.png"
        outpath = os.path.join(output_dir, outname)

        Image.fromarray(mask).save(outpath)

        print(f"[{i}/{len(images)}] Saved {outpath}")

    print("\n✔ All masks saved!")


if __name__ == "__main__":
    WEIGHTS = "../SEGMENTATION/weights/unet_legacy.pt"
    INPUT_DIR = "../SEG_ON_SYN/imgs"
    OUTPUT_DIR = "../SEG_ON_SYN/out"

    model = load_model(WEIGHTS)
    process_directory(INPUT_DIR, OUTPUT_DIR, model)
