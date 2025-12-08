import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from globals import IMAGE_SIZE

crop_and_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.NEAREST),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.Grayscale(num_output_channels=1),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Invert the normalization process
invert_normalization = transforms.Compose([
    transforms.Normalize(mean=[-1], std=[2])
])

# Same output format, without normalization
crop_and_resize = transforms.Compose([
    crop_and_normalize, 
    invert_normalization
])
