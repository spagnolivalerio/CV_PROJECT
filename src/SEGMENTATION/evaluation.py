import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from dataset import DentalDataset
from globals import DEVICE, NUM_CLASSES, DATA_ROOT
from utils import tensor_to_image, color_mask, classes_to_palette, make_overlay, plot_results
from metrics import multiclass_dice, multiclass_iou, multiclass_precision_recall, pixel_accuracy

WEIGHT_PATH = "weights/unet_legacy.pt"
IMGS_DIR  = DATA_ROOT + "/validation_set/xrays"
MASKS_DIR = DATA_ROOT + "/validation_set/masks"
BATCH_SIZE = 10

# Loading the model
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=NUM_CLASSES,
).to(DEVICE)

# Loading the weights
model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
model.eval()

dataset = DentalDataset(IMGS_DIR, MASKS_DIR, augment=False)
eval_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
printing_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Printing the evaluation metrics
with torch.no_grad():

    val_miou = 0
    val_mdice = 0
    val_prec = 0
    val_rec  = 0
    val_acc  = 0

    for img, mask in eval_dataloader:

        img = img.to(DEVICE)
        mask = mask.to(DEVICE)
        pred = model(img)

        miou = multiclass_iou(pred, mask, NUM_CLASSES)
        mdice = multiclass_dice(pred, mask, NUM_CLASSES)
        precs, recs = multiclass_precision_recall(pred, mask, NUM_CLASSES)
        acc = pixel_accuracy(pred, mask)

        val_miou += miou
        val_mdice += mdice
        val_prec += np.nanmean(precs)
        val_rec  += np.nanmean(recs)
        val_acc  += acc
    
    val_miou  /= len(eval_dataloader)
    val_mdice /= len(eval_dataloader)
    val_prec  /= len(eval_dataloader)
    val_rec   /= len(eval_dataloader)
    val_acc   /= len(eval_dataloader)
    
    print(
        f"  mIoU: {val_miou:.4f} "
        f"| Mean Dice(F1): {val_mdice:.4f} "
        f"| Precision: {val_prec:.4f} "
        f"| Recall: {val_rec:.4f} "
        f"| PixelAcc: {val_acc:.4f}"
    )

# Building the palette and plotting the results
palette = classes_to_palette()
with torch.no_grad():

    for i, (img, mask) in enumerate(printing_dataloader):

        pred = model(img.to(DEVICE))
        pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
        mask = mask.squeeze().cpu().numpy()
        original_img = tensor_to_image(img)

        colored_mask = color_mask(pred, palette)
        colored_gt = color_mask(mask, palette)
        overlay  = make_overlay(original_img, colored_mask)

        plot_results(original_img, colored_gt, colored_mask, overlay)
        break

