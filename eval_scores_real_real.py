import os
import shutil
import tempfile
import numpy as np
from itertools import combinations

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss
from tqdm import tqdm

from monai.transforms import (
    EnsureChannelFirstd,
    CenterSpatialCropd,
    Compose,
    Lambdad,
    LoadImaged,
    Resized,
    ScaleIntensityd,
)
from monai.utils import set_determinism
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from monai.data import (
    CacheDataset,
    Dataset,
    DataLoader,
    LMDBDataset,
    PersistentDataset,
    decollate_batch,
)

from generative.inferers import LatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler
from generative.metrics import FIDMetric, MMDMetric, MultiScaleSSIMMetric, SSIMMetric

GPU_NUM = '1'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_NUM
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
print("Current GPU:",  os.environ["CUDA_VISIBLE_DEVICES"])

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

device = torch.device(f"cuda:{GPU_NUM}" if torch.cuda.is_available() else "cpu") #ADDED LATER

ms_ssim = MultiScaleSSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=2)
ssim = SSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=2)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".npy"):
            img_path = os.path.join(folder, filename)
            img = np.load(img_path)
            images.append(img)
    return images

dataset_name = "CAMCAN"
dataset_path = "../../kozarkar/Diffusion/medicaldiffusion/data/CAMCAN/images/MALE" 
os.environ["MONAI_DATA_DIRECTORY"]= dataset_path
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

data_transform = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        ScaleIntensityd(keys=["image"]),
        CenterSpatialCropd(keys=["image"], roi_size=[160, 200, 160]),
        Resized(keys=["image"], spatial_size=(64, 64, 64)),
    ]
)

def get_data_files(root_dir):
    nifti_file_names = os.listdir(root_dir)
    folder_names = [os.path.join(
        root_dir, nifti_file_name) for nifti_file_name in nifti_file_names if nifti_file_name.endswith('.nii.gz')]
    return folder_names

paths = get_data_files(root_dir)
train_files = [{"image": path} for path in paths]
len(train_files)

train_dataset = CacheDataset(
    data=train_files, transform=data_transform, cache_rate=1.0, runtime_cache="processes", copy_cache=False 
)

train_loader = DataLoader(train_dataset, batch_size=1, num_workers=10, persistent_workers=True)

male_real_images = []

for batch in train_loader:
    male_real_images.append(batch['image'])

male_real_images_tensor = torch.stack(male_real_images)

print(male_real_images_tensor.shape[0])


ms_ssim_scores = []
ssim_scores = []

idx_pairs = list(combinations(range(male_real_images_tensor.shape[0]), 2))
print(len(idx_pairs))

print("Diversity (Real vs Real) MS-SSIM")

for idx_a, idx_b in tqdm(idx_pairs, desc="Calculating SSIM and MS-SSIM", unit="pair"):
    img_a = male_real_images_tensor[idx_a]  
    img_b = male_real_images_tensor[idx_b]  
    ms_ssim_scores.append(ms_ssim(img_a, img_b))
    ssim_scores.append(ssim(img_a, img_b))

ms_ssim_scores = torch.cat(ms_ssim_scores, dim=0)
ssim_scores = torch.cat(ssim_scores, dim=0)
print(f"MS-SSIM Metric: {ms_ssim_scores.mean():.4f} +- {ms_ssim_scores.std():.4f}")
print(f"SSIM Metric: {ssim_scores.mean():.4f} +- {ssim_scores.std():.4f}")