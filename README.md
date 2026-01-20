# Computer Vision Project - Dental X-Ray Images Generation and Segmentation
**Author:** Spagnoli Valerio (1973484)

## Project outline
This project focuses on dental X-ray image generation and multi-class segmentation. It includes:
- synthetic image generation with WGAN-GP and Diffusion model (DDPM);
- supervised segmentation with U-Net (ResNet34 encoder);
- SYNSEG variant: training the segmenter using hybrid dataset (synthetic images + real images).

## Project goals
- generate realistic dental X-rays;
- improve segmentation performance using synthetic data;
- quantify generation quality with FID and Inception Score;
- evaluate segmentation with standard multi-class metrics.

## Dataset composition
This project combines two sources:
- an annotated dataset from Kaggle (with pixel-wise masks);
- an unannotated dataset from Dentex (images only).

All images from both datasets are merged to create a hybrid pool used for generation. For segmentation, only the Kaggle dataset with annotations is used. This segmentation set is further strengthened by adding synthetic images that are pseudo-labeled by a segmentation model trained only on the Kaggle annotations.

**Dataset references:**
- https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images
- https://dentex.grand-challenge.org/data/

## Common section structure
Each module follows a consistent layout:
> - `globals.py`: hyperparameters, paths, device, image size.
> - `dataset.py`: dataset loading and preprocessing.
> - `utils.py`: transforms, normalization, visualization helpers.v
> - `network.py`: model architecture (GAN/DDPM).
> - `train.py`: training loop and checkpointing.
> - `sampler.py`: sampling/generation from trained models (when present).
> - `evaluation.py`: evaluation scripts and metrics (when present).
> - `metrics.py`: metric implementations (segmentation and generative).

## Project sections

### `src/SEGMENTATION/` (supervised segmentation)
> **Purpose:** perform semantic segmentation on provided dataset.

**Model:** U-Net with `resnet34` encoder (_**ImageNet pretrained**_), input grayscale, output 33 classes (32 teeth + background):
```
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=NUM_CLASSES,
)
```
**Training:** `train.py` trains using CrossEntropyLoss on augmented dataset;
**Evaluation:** `evaluation.py` prints mIoU, Dice, Precision, Recall, Pixel Accuracy and saves an overlay example.

### `src/SYNSEG/` (segmentation on synthetic data)
> **Purpose:** measure whether synthetic data improves segmentation.

**Model:** same U-Net configuration as segmentation.  
**Training:** uses synthetic images and masks (`diff_xrays`, `diff_masks`) for training. Different combinations of synthetic and real images have been used:
- `ddpm syn images` + `real images`
- `wgan-gp syn images` + `real images`
- `ddpm syn images`

### `src/WGAN-GP/` (generation with GAN)
> **Purpose:** generate synthetic data.

**Model:** WGAN-GP with a ConvTranspose2d generator and Conv2d critic (LeakyReLU) built from scratch (implementation in `src/WGAN-GP/network.py`).  
**Loss:** Wasserstein with Gradient Penalty (implementation in `src/WGAN-GP/utils.py` - `gradient_penalty()`).  
**Training:** `train.py` alternates critic and generator steps, saving samples to track convergence.  
**Sampling:** `sampler.py` generates images into `data/fake`.  
**Evaluation:** `evaluation.py` computes FID and Inception Score.

### `src/DIFFUSION/` (generation with DDPM)
> **Purpose:** generate synthetic data.


**Model:** `UNet2DModel` from `diffusers`:
```
model = UNet2DModel(
            sample_size=image_size,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(32, 64, 128), 
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        )
```  
**Noise schedule:** linear betas from `BETA_START` to `BETA_END` (implementation in `src/DIFFUSION/network.py`).  
**EMA:** Exponential Moving Average for stability and higher-quality sampling (implementation in `src/DIFFUSION/network.py`), better convergence wrt DDPM without EMA.   
**Training:** `train.py` asks for EMA (`1`) or classic (`0`) mode, logs loss, saves checkpoints.  
**Sampling:** `sampler.py` loads an EMA checkpoint and generates samples.
**Notes**: DDPM training requires a lot of time, so a checkpointing mechanism has been introduced. Checkpoint data structure:
```
ckpt = {
        "model_state_dict": diffusion.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step
        }
```

## Preprocessing
### Images
#### Transformations
- grayscale conversion, resize to `IMAGE_SIZE`, center crop;
- normalization to `[-1, 1]` for WGAN/DDPM and segmentation.
```
crop_and_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.BILINEAR),
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
```
#### Data augmentation

Data used for the segmentation task were augmented using two types of augmentation:
> - horizontal flip 
> - rotation

randomly adopted during the training:
```
# Online augmentation
if self.augment:

            if random.random() < 0.5:

                # Horizontal flip of the image (PIL img, numpy mask)
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = np.fliplr(mask)

            if random.random() < 0.5:
                angle = random.uniform(-90, 90)

                # Rotate image (BILINEAR is good, no discrete indexes)
                img = img.rotate(angle, resample=Image.BILINEAR)

                # rotate mask (NEAREST is good for discrete values)
                mask_pil = Image.fromarray(mask)
                mask_pil = mask_pil.rotate(angle, resample=Image.NEAREST)
                mask = np.array(mask_pil)
```
### Masks
- resize and center crop with nearest-neighbor interpolation to preserve labels.
```
mask_crop = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.NEAREST),
    transforms.CenterCrop(IMAGE_SIZE)
])
```

### Annotation conversion
`src/SEGMENTATION/data/training_set/convert.py` converts JSON annotations to PNG masks using `classId -> index` mapping from `meta.json` (related to kaggle dataset).

## Evaluation metrics 

### Segmentation metrics
**Metrics**: mIoU, Dice, Precision, Recall, Pixel Accuracy. Implementation in `src/SEGMENTATION/metrics.py`.  

### Generative metrics
**Metrics**: FID (clean-fid) and Inception Score. Implementation in `src/DIFFUSION/metrics.py`, `src/WGAN-GP/metrics.py`  

## Experimental setup

### DDPM with EMA (Diffusion)
| Hyperparameter | Value |
| --- | --- |
| Image size | 128 |
| Batch size | 16 |
| Timesteps | 450 |
| Beta schedule | 1e-4 to 0.02 (linear) |
| Total steps (EMA) | 100000 |
| Learning rate (EMA) | 8e-5 |
| EMA decay | 0.999 |
| Optimizer | Adam |

### WGAN-GP
| Hyperparameter | Value |
| --- | --- |
| Image size | 128 |
| Latent dim (Z) | 100 |
| Generator channels | 64 |
| Critic channels | 64 |
| Batch size | 16 |
| Epochs | 1000 |
| Critic steps | 2 |
| Generator LR | 1e-4 |
| Critic LR | 5e-5 |
| Gradient penalty lambda | 7 |
| Optimizer | Adam (betas 0.0, 0.9) |

### Segmentation (U-Net)
| Hyperparameter | Value |
| --- | --- |
| Image size | 128 |
| Batch size | 10 |
| Epochs | 100 |
| Learning rate | 1e-4 |
| Encoder | ResNet34 (ImageNet pretrained) |
| Classes | 33 |
| Optimizer | Adam |

## Notes
- Default device is `cuda`. To run on CPU, edit `DEVICE` in each `globals.py`.
- FID and IS require RGB conversion from grayscale; the scripts handle it.
