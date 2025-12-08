import torchvision.transforms as transforms
import numpy as np
from torchvision.transforms import InterpolationMode
from globals import IMAGE_SIZE, NUM_CLASSES
import matplotlib.pyplot as plt

mask_crop = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.NEAREST),
    transforms.CenterCrop(IMAGE_SIZE)
])

crop_and_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.NEAREST),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.Grayscale(num_output_channels=1),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Build a color palette starting from a list of class indexes
def classes_to_palette():
    PALETTE = []
    for i in range(NUM_CLASSES):
        np.random.seed(i)
        PALETTE.append(np.random.randint(0, 255, 3).tolist())
    return PALETTE

# Transform a tensor to an image, performing de-normalization
def tensor_to_image(t):
    t = t.squeeze().cpu().numpy()
    t = (t * 0.5 + 0.5)        
    return (t * 255).clip(0,255).astype(np.uint8)

# Color the mask (Background (class 0) is ignored)
def color_mask(mask, palette):

    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    for cls, color in enumerate(palette):
        if cls == 0:
            continue 
        out[mask == cls] = color

    return out

# Perform the mask overlay over the image
def make_overlay(img, mask, alpha=0.5):
    img = img.astype(np.float32) / 255
    img_3channel = np.stack([img, img, img], axis=-1)
    mask = mask.astype(np.float32) / 255

    # Blending the image
    return (1 - alpha) * img_3channel + alpha * mask

# Plot 1 row with 4 columns ==> (img, gt, pred, overlay)
def plot_results(original_img, colored_gt, colored_mask, overlay):
    plt.figure(figsize=(15,4))

    plt.subplot(1,4,1)
    plt.imshow(original_img, cmap="gray")
    plt.title("ORIGINAL IMAGE")
    plt.axis("off")

    plt.subplot(1,4,2)
    plt.imshow(colored_gt)
    plt.title("GROUND TRUTH")
    plt.axis("off")

    plt.subplot(1,4,3)
    plt.imshow(colored_mask)
    plt.title("PREDICTION")
    plt.axis("off")

    plt.subplot(1,4,4)
    plt.imshow(overlay)
    plt.title("OVERLAY")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
