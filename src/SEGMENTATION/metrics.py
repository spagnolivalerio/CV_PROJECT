import numpy as np

# Avoid division by 0, stabilizing the division
EPS = 1e-6

def logits_to_class(preds):
    # preds shape: (B, C, H, W)
    # Take highest value on dim=1 => C (highest probability)
    return preds.argmax(dim=1)

# Multiclass version of the union over intersection
def multiclass_iou(preds, target, num_classes):
    preds = logits_to_class(preds)

    ious = []

    for c in range(num_classes):
        pred_inds = (preds == c)
        target_inds = (target == c)

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            ious.append(np.nan) # Numpy null value
        else:
            ious.append(intersection / (union + EPS))

    # nanmean doesn't take care about nan values
    mean_iou = np.nanmean(ious)
    return mean_iou

# Multiclass version of dice score
def multiclass_dice(preds, target, num_classes):
    preds = logits_to_class(preds)

    dice_scores = []

    for c in range(num_classes):
        current_pred = (preds == c)
        current_target = (target == c)

        tp = (current_pred & current_target).sum().item()
        fp = (current_pred & ~current_target).sum().item()
        fn = (~current_pred & current_target).sum().item()

        denom = (2 * tp + fp + fn)

        if denom == 0:
            dice_scores.append(np.nan)
        else:
            dice_scores.append((2 * tp) / (denom + EPS))

    mean_dice = np.nanmean(dice_scores)
    return mean_dice

# Multiclass version of precision and recall
def multiclass_precision_recall(preds, target, num_classes):
    preds = logits_to_class(preds)

    precisions = []
    recalls = []

    for c in range(num_classes):
        current_pred = (preds == c)
        current_target = (target == c)

        tp = (current_pred & current_target).sum().item()
        fp = (current_pred & ~current_target).sum().item()
        fn = (~current_pred & current_target).sum().item()

        if (tp + fp) == 0:
            precisions.append(np.nan)
        else:
            precisions.append(tp / (tp + fp + EPS))

        if (tp + fn) == 0:
            recalls.append(np.nan)
        else:
            recalls.append(tp / (tp + fn + EPS))

    return precisions, recalls

# Accuracy
def pixel_accuracy(preds, target):
    preds = logits_to_class(preds)
    correct = (preds == target).sum().item()
    total = preds.numel() # Computes the product of the dimension of the tensor
    return correct / total
