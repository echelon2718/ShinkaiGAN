import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm.auto import tqdm
import torchvision
from torchvision import transforms
from torchvision.models import vgg16
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from IPython.display import clear_output, display

from module.ProHPBUNet import Generator, Discriminator
from module.loss_fn import *
from module.dataset import download_dataset
from module.trainer import Trainer, weights_init
from data_util.dataloader import MakotoShinkaiDataset

parser = argparse.ArgumentParser(description='Train ShinkaiGAN on Makoto Shinkai dataset.')
parser.add_argument('--src_dir', type=str, required=True, help='Path to source directory')
parser.add_argument('--tgt_dir', type=str, required=True, help='Path to target directory')
parser.add_argument('--lvl1_epoch', type=int, required=True, help='Number of epochs for level 1 training')
parser.add_argument('--lvl2_epoch', type=int, required=True, help='Number of epochs for level 2 training')
parser.add_argument('--lvl3_epoch', type=int, required=True, help='Number of epochs for level 3 training')
parser.add_argument('--lvl4_epoch', type=int, required=True, help='Number of epochs for level 4 training')
parser.add_argument('--lambda_adv', type=float, default=1, help='Weight for adversarial loss (default: 1)')
parser.add_argument('--lambda_ct', type=float, default=0.1, help='Weight for content loss (default: 0.1)')
parser.add_argument('--lambda_up', type=float, default=0.01, help='Weight for upper loss (default: 0.01)')
parser.add_argument('--lambda_style', type=float, default=0.01, help='Weight for style loss (default: 0.01)')
parser.add_argument('--lambda_color', type=float, default=0.001, help='Weight for color constancy loss (default: 0.001)')
parser.add_argument('--lambda_grayscale', type=float, default=0.01, help='Weight for grayscale loss (default: 0.01)')
parser.add_argument('--lambda_tv', type=float, default=0.001, help='Weight for total variation loss (default: 0.001)')
parser.add_argument('--lambda_fml', type=float, default=0.01, help='Weight for feature matching loss (default: 0.01)')
parser.add_argument('--device', type=str, default="cuda", help='Train model using CUDA GPU or CPU (default: cuda)')

args = parser.parse_args()

download_dataset()

try:
    device = args.device
except:
    device = "cuda" if torch.cuda.is_available() else "cpu"

gen = Generator().to(device)
disc = Discriminator().to(device)

gen = gen.apply(weights_init)
disc = disc.apply(weights_init)

adv_loss = AdversarialLoss()
content_loss = PerceptualLoss(vgg16(weights="DEFAULT").to(device), type_loss="content")
upper_loss = PerceptualLoss(vgg16(weights="DEFAULT").to(device))
style_loss = PerceptualLoss(vgg16(weights="DEFAULT").to(device), type_loss="style")
color_const = ColorLoss(nc=3).to(device)
grayscale_loss = GrayscaleLoss()
tv_loss = TotalVariationLoss()
feat_match_loss = FeatureMatchingLoss()

gen_loss = GeneratorLoss(adv_loss, 
                         content_loss, 
                         upper_loss, 
                         style_loss, 
                         color_const,
                         grayscale_loss,
                         tv_loss,
                         feat_match_loss,
                         
                         lambda_adv=args.lambda_adv, 
                         lambda_ct=args.lambda_ct,
                         lambda_up=args.lambda_up, 
                         lambda_style=args.lambda_style,
                         lambda_color=args.lambda_color, 
                         lambda_grayscale=args.lambda_grayscale,
                         lambda_tv=args.lambda_tv,
                         lambda_fml=args.lambda_fml)

disc_loss = DiscriminatorLoss()

gen = gen.apply(weights_init)
disc = disc.apply(weights_init)

trainer = Trainer(gen, disc, gen_loss, disc_loss, 8)
trainer.progressive_training(epochs=[args.lvl1_epoch, args.lvl2_epoch, args.lvl3_epoch, args.lvl4_epoch], levels=[2, 5, 8, 15], src_dir=args.src_dir, tgt_dir=args.tgt_dir)