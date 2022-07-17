"""
:author: anton

This script is used only as a benchmark tool for different batch sizes and learning rates.
"""

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchgeometry.losses import ssim

from dataset.dataset import StereoDataset
from src.loss import calculate_lr_consistency
from src.model import Autoencoder
from utils.bilinear_sampler import get_interpolated_images

writer = SummaryWriter("../runs/benchmark")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device used:", device)

# Load dataset
train_dataset = StereoDataset("../dataset/train.pt")

dataset_length = len(train_dataset)
validation_length = int(np.floor(dataset_length * 0.2))

train_subset, val_subset = random_split(train_dataset, [dataset_length - validation_length, validation_length],
                                        generator=torch.Generator().manual_seed(1))

best_val_loss = 1_000.0

alpha_l = alpha_r = 0.5
alpha_c = 1.0

global_losses = {}

for batch in [8, 16, 25]:
    for lr in [10e-3, 10e-4, 10e-5]:

        train_loader = DataLoader(dataset=train_subset, batch_size=batch, shuffle=True)
        val_loader = DataLoader(dataset=val_subset, shuffle=True, batch_size=batch)
        # test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size)

        # Init the network
        model = Autoencoder().to(device)

        # Set the optimizer and lr scheduler
        optimizer = optim.Adam(model.parameters(), lr=lr)

        print("\nBenchmark ==>")

        # Iterate over all batches in training dataset
        for batch_idx, (left_img, right_img) in enumerate(train_loader):
            left_img = left_img.to(device=device)
            right_img = right_img.to(device=device)

            # feed the network with the data
            dl = model(left_img)
            dr = model(right_img)

            # reconstruct estimated maps back to inputs
            i_left = get_interpolated_images(right_img, -dl)
            i_right = get_interpolated_images(left_img, dr)

            # calculate losses
            l_loss = torch.mean(torch.abs(left_img - i_left))
            r_loss = torch.mean(torch.abs(right_img - i_right))
            c_loss = calculate_lr_consistency(dl, dr)

            loss = alpha_l * l_loss + alpha_r * r_loss + alpha_c * c_loss

            # for metrics -- Structural similarity was used (SSIM)
            l_metrics = 1 - ssim(i_left, left_img, window_size=3, reduction='mean')
            r_metrics = 1 - ssim(i_right, right_img, window_size=3, reduction='mean')

            # perform the backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalars("Benchmark loss", {
                f'Loss {batch}/{lr}': loss.item()
            }, batch_idx)

writer.close()
