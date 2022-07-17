import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# lc = 1/N * SUM_{i,j}(Dl(i,j) - Dr(i + Dl(i,j), j)


def calculate_lr_consistency(dl, dr):
    batch = dl.size(0)
    height = dl.size(2)
    width = dl.size(3)

    i = torch.arange(height, device=device).view(1, 1, height, 1).repeat(batch, 1, 1, width)
    disparity_indices = torch.clamp((i + dl).round(), max=height - 1).int()
    lr_loss = torch.mean(torch.abs(dl - torch.gather(dr, 2, disparity_indices.type(torch.int64))))

    return lr_loss
