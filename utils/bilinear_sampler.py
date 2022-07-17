import torch
from torch.nn import functional as F


def get_interpolated_images(images, disps):
    N, C, H, W = images.size()

    mesh_x, mesh_y = torch.meshgrid(torch.linspace(0, 1, W),
                                    torch.linspace(0, 1, H),
                                    indexing='xy')

    mesh_x = mesh_x.repeat(N, 1, 1).type_as(images)
    mesh_y = mesh_y.repeat(N, 1, 1).type_as(images)

    grid = torch.stack((mesh_x + disps.squeeze(), mesh_y), 3)

    grid = 2 * grid - 1

    output = F.grid_sample(images, grid, mode='bilinear', padding_mode='zeros')
    return output
