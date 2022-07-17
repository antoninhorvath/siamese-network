"""
:author: anton

This script implements Siamese architecture for generating disparity maps. Two images are fed into the autoencoder,
generating two outputs compared by the function lc. Separately, those images are given to spatial transformer blocks
respectively along their inverse counterparts. Results of these operations are compared with the input images by losses
l_right and l_left.
"""

import os.path

import numpy as np
import torch
import torch.optim as optim
import torchvision
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchgeometry.losses import ssim

from dataset.dataset import StereoDataset
from loss import calculate_lr_consistency
from model import Autoencoder
from utils.bilinear_sampler import get_interpolated_images

writer = SummaryWriter("../runs/train_siamese_64")
load_model = False
checkpoint_path = '../model_checkpoints/checkpoint_64.pt'
dataset_path = "../dataset/train_64.pt"

if not os.path.isdir(checkpoint_path) and not os.path.isfile(dataset_path):
    print("Resources (checkpoint directory or dataset file) not found")
    print("Current dir:", os.getcwd())
    exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device used:", device)

# Hyperparams
learning_rate = 10e-4
batch_size = 16
num_epochs = 40

# Load dataset
train_dataset = StereoDataset(dataset_path)

dataset_length = len(train_dataset)
validation_length = int(np.floor(dataset_length * 0.2))

train_subset, val_subset = random_split(train_dataset, [dataset_length - validation_length, validation_length])

train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_subset, shuffle=True, batch_size=batch_size)

# Init the network
model = Autoencoder().to(device)

# Set the optimizer and lr scheduler
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# when training is interrupted, set load_model to True, so the training
# can continue where it has stopped
if load_model:
    cp = torch.load(checkpoint_path)
    model.load_state_dict(cp['model'])
    optimizer.load_state_dict(cp['optimizer'])
    scheduler.load_state_dict(cp['scheduler'])

alpha_l = alpha_r = 0.5
alpha_c = 1.0

early_stop_step = 0
patience = 3
best_val_loss = 1_000.0


def train(epoch):
    avg_train_loss = 0.0
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
        avg_train_loss += loss.item()

        # for metrics -- Structural similarity was used (SSIM)
        l_metrics = 1 - ssim(i_left, left_img, window_size=3, reduction='mean')
        r_metrics = 1 - ssim(i_right, right_img, window_size=3, reduction='mean')

        # perform the backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("Loss train", loss.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalars("Metrics train",
                           {'Left': l_metrics.item(), 'Right': r_metrics.item()},
                           epoch * len(train_loader) + batch_idx)

        # tensorboard visualization for each 500th batch
        if batch_idx % 500 == 0:
            with torch.no_grad():
                grid_l = torchvision.utils.make_grid(dl)
                grid_r = torchvision.utils.make_grid(dr)

                writer.add_image(f"Left disparity {batch_idx}", grid_l, epoch + 1)
                writer.add_image(f"Right disparity {batch_idx}", grid_r, epoch + 1)
        writer.flush()

    writer.add_scalars("Train/Val losses", {
        "Train": avg_train_loss / len(train_loader)
    }, epoch)

    scheduler.step()


def validate(epoch):
    global early_stop_step
    global best_val_loss
    with torch.no_grad():

        if early_stop_step == patience:
            return 1

        avg_loss = 0.0
        for batch_idx, (left_img, right_img) in enumerate(val_loader):
            left_img = left_img.to(device)
            right_img = right_img.to(device)

            dl = model(left_img)
            dr = model(right_img)

            i_left = get_interpolated_images(right_img, -dl)
            i_right = get_interpolated_images(left_img, dr)

            l_loss = torch.mean(torch.abs(left_img - i_left))
            r_loss = torch.mean(torch.abs(right_img - i_right))
            c_loss = calculate_lr_consistency(dl, dr)

            vloss = alpha_l * l_loss + alpha_r * r_loss + alpha_c * c_loss
            avg_loss += vloss.item()

            writer.add_scalar("Loss validate", vloss.item(), epoch * len(val_loader) + batch_idx)
            writer.flush()

        print(f"\n\tAverage validation loss for epoch {epoch} is {avg_loss / len(val_loader)}")

        avg_loss = avg_loss / len(val_loader)

        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            early_stop_step = 0
        else:
            early_stop_step += 1

        writer.add_scalars("Train/Val losses", {
            "Validation": avg_loss
        }, epoch)

    return 0


def main():
    for epoch in range(num_epochs):

        train(epoch)
        code = validate(epoch)

        if code == 1:
            break

        print("Saving the model")
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, checkpoint_path)


if __name__ == '__main__':
    main()
    writer.close()
