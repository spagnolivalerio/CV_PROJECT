import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from utils import crop_and_normalize

class DentalDataset(Dataset):
    def __init__(self, root_dir):
        self.paths = sorted(glob(os.path.join(root_dir, "*")))
        if len(self.paths) == 0:
            print(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        img = crop_and_normalize(img)
        return img  