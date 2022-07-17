"""
:author: anton

This script implements basic architecture for disparity map generation.
Input left images are fed into the autoencoder then into the spatial transformer along with right images.
Result of this operation is compared with the initial left input image via the loss function - criterion.
"""
import numpy as np
import torch
import torch.optim as optim
import torchvision
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# when using conda - use 'pip install torchgeometry' command since conda 
# doesn't have any version for this package available.
from torchgeometry.losses import ssim

from dataset.dataset import StereoDataset
from model import Autoencoder
from utils.bilinear_sampler import get_interpolated_images


def criterion(prediction, target):
    return torch.mean(torch.abs(target - prediction))


writer = SummaryWriter("runs/train_siamese_big")
load_model = False
checkpoint_path = './dataset/checkpoint_siamese_big_dataset.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Hyperparams
learning_rate = 10e-4
batch_size = 16
num_epochs = 1
initial_epoch = 0

# Load dataset
train_dataset = StereoDataset("./dataset/train_big.pt")
# test_dataset = StereoDataset("./dataset/test.pt", device)

dataset_length = len(train_dataset)
validation_length = int(np.floor(dataset_length * 0.2))

train_subset, val_subset = random_split(train_dataset, [dataset_length - validation_length, validation_length],
                                        generator=torch.Generator().manual_seed(1))

train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_subset, shuffle=True, batch_size=batch_size)
# test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size)

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

best_val_loss = 1_000.0

for epoch in range(num_epochs):
    print("\nTraining ==>")

    train_losses = 0.0
    val_losses = 0.0

    train_metrics = 0.0
    val_metrics = 0.0

    # Iterate over all batches in training dataset
    for batch_idx, (left_img, right_img) in enumerate(train_loader):
        left_img = left_img.to(device=device)
        right_img = right_img.to(device=device)

        estimate = model(left_img)
        reconstructed = get_interpolated_images(right_img, -estimate)

        output_loss = criterion(reconstructed, left_img)

        optimizer.zero_grad()
        output_loss.backward()
        optimizer.step()

        train_losses += output_loss.item()

        current_metric = 1 - ssim(left_img, reconstructed, window_size=3, reduction='mean')
        train_metrics += current_metric.item()

        writer.add_scalar("Train loss", output_loss.item(), (epoch + initial_epoch) * len(train_loader) + batch_idx)
        writer.add_scalar("Train metrics", current_metric.item(), (epoch + initial_epoch) * len(train_loader) + batch_idx)

        if (batch_idx + 1) % 500 == 0:
            with torch.no_grad():
                # print('\n', torch.min(estimate[0]), torch.max(estimate[0]))
                grid = torchvision.utils.make_grid(estimate)
                writer.add_image("Estimated", grid, 0)

    # Increment the scheduler
    scheduler.step()

    print("\nEvaluating ===>")

    # VALIDATE
    with torch.no_grad():
        for batch_idx, (left_imgs, right_imgs) in enumerate(val_loader):
            left_imgs = left_imgs.to(device)
            right_imgs = right_imgs.to(device)

            estimate = model(left_imgs)
            reconstructed = get_interpolated_images(right_imgs, -estimate)

            output_loss = criterion(reconstructed, left_imgs)
            val_losses += output_loss.item()

            current_metric = 1 - ssim(left_imgs, reconstructed, window_size=3, reduction='mean')
            val_metrics += current_metric.item()

            writer.add_scalar("Validation loss", output_loss.item(), (epoch + initial_epoch) * len(val_loader) + batch_idx)
            writer.add_scalar("Validation metrics", current_metric.item(), (epoch + initial_epoch) * len(val_loader) + batch_idx)

            # loop.set_description(f"EPOCH [{(epoch + initial_epoch)}/{num_epochs}]")
            # loop.set_postfix(loss=output_loss.item(), metrics=current_metric.item())

    avg_train_loss = train_losses / len(train_loader)
    avg_validation_loss = val_losses / len(val_loader)

    avg_train_metrics = train_metrics / len(train_loader)
    avg_validation_metrics = val_metrics / len(val_loader)

    # print(f"LOSS train {avg_train_loss} valid {avg_validation_loss}")

    writer.add_scalars("Average Training vs. Validation loss",
                       {
                           'Training': avg_train_loss,
                           'Validation': avg_validation_loss
                       }, (epoch + initial_epoch) + 1)

    writer.add_scalars("Average Training vs. Validation metrics",
                       {
                           'Training': avg_train_metrics,
                           'Validation': avg_validation_metrics
                       }, (epoch + initial_epoch) + 1)

    writer.flush()

    if avg_validation_loss < best_val_loss:
        best_val_loss = avg_validation_loss
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, checkpoint_path)
