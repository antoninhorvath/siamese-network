"""
:author: anton

Testing loop for the trained siamese model.
"""
import os

import torch
import torchgeometry
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.dataset import StereoDataset
from model import Autoencoder
from utils.bilinear_sampler import get_interpolated_images
from utils.visualize import show_disparities

writer = SummaryWriter("./runs/test_64")

BATCH_SIZE = 16

device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_path = "./checkpoint_tmp_32.pt"
dataset_path = "./dataset/test.pt"

test_dataset = StereoDataset(dataset_path)
test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=BATCH_SIZE)

model = Autoencoder().to(device)

checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model'])

visualization_disparities = False
visualization = False

model.eval()
with torch.no_grad():
    similarity_score_l = 0.0
    similarity_score_r = 0.0

    for batch_idx, (left_img, right_img) in enumerate(test_loader):
        row = 0

        left_img = left_img.to(device)
        right_img = right_img.to(device)

        disparity_left = model(left_img)
        disparity_right = model(right_img)
        reconstructed_left = get_interpolated_images(right_img, -disparity_left)
        reconstructed_right = get_interpolated_images(left_img, disparity_right)

        similarity_left = 1 - torchgeometry.losses.ssim(reconstructed_left, left_img, window_size=3, reduction='mean')
        similarity_right = 1 - torchgeometry.losses.ssim(reconstructed_right, right_img, window_size=3, reduction='mean')

        similarity_score_l += similarity_left
        similarity_score_r += similarity_right

        writer.add_scalar("test", similarity_left, batch_idx)

        if visualization_disparities:
            show_disparities(left_img, right_img, disparity_left)
            break

        if batch_idx % 20 == 0 and visualization:
            d_l = disparity_left[1].cpu().permute(1, 2, 0)
            d_r = disparity_right[1].cpu().permute(1, 2, 0)
            color_image_left = left_img[1].cpu().permute(1, 2, 0)
            color_image_right = right_img[1].cpu().permute(1, 2, 0)
            r_left = reconstructed_left[1].cpu().permute(1, 2, 0)
            r_right = reconstructed_right[1].cpu().permute(1, 2, 0)

            fig = plt.figure()
            axes = []
            titles = ['Levý vstup', 'Pravý vstup', 'Levá interpolace', 'Pravá interpolace',
                      'Levá disparitní mapa', 'Pravá disparitní mapa']
            for i in range(6):
                ax = fig.add_subplot(321 + i)
                ax.title.set_text(titles[i])
                ax.axis('off')
                axes.append(ax)

            axes[0].imshow(color_image_left)
            axes[1].imshow(color_image_right)
            axes[2].imshow(r_left)
            axes[3].imshow(r_right)
            axes[4].imshow(d_l, cmap='gray')
            axes[5].imshow(d_r, cmap='gray')

            plt.show()
            plt.pause(0.001)

    print(f"Avg similarity left: {similarity_score_l / len(test_loader)}")
    print(f"Avg similarity right: {similarity_score_r / len(test_loader)}")




