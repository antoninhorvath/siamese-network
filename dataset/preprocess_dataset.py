"""
Module for image preprocessing

:author: anton

:cite: Dataset used for this project: Driving stereo large scale dataset https://drivingstereo-dataset.github.io/

Dataset structure in folder:

dataset/
    |____test/ \n\t
        |______ left/ \n\t
        |______ right/ \n
    |____train/
        |______ left/
        |______ right/

    |____<train_[32|64]>.pt     -- generated after running this script

"""
from glob import glob

import numpy as np
import torch
from PIL import Image
from typing import List, Tuple

IMG_WIDTH = 64
IMG_HEIGHT = 64




def save_data():
    """
    Image preprocessing for training, images are saved to torch tensor.

    raises: FileNotFound exception
    """

    left_train = sorted(glob('./train/left/*'))
    right_train = sorted(glob('./train/right/*'))

    left_test = sorted(glob('./test/left/*'))
    right_test = sorted(glob('./test/right/*'))

    train_count = len(left_train)
    test_count = len(left_test)

    print("\nProcessing...")
    train_images = convert_to_array(train_count, left_train, right_train)
    test_images = convert_to_array(test_count, left_test, right_test)

    print("\nSaving to disk...")
    torch.save({
        'left': torch.from_numpy(train_images[0]).permute(0, 3, 1, 2),
        'right': torch.from_numpy(train_images[1]).permute(0, 3, 1, 2)
    }, './train_64.pt')

    torch.save({
        'left': torch.from_numpy(test_images[0]).permute(0, 3, 1, 2),
        'right': torch.from_numpy(test_images[1]).permute(0, 3, 1, 2)
    }, './test_64.pt')

    # np.savez(dataset_path, train=train_images, test=test_images)


def convert_to_array(img_count: int,
                     left_images: List[str],
                     right_images: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    stereo_pairs = {
        'left': [],
        'right': []
    }

    for i in range(img_count):
        print(f"\r{int(i / img_count * 100)}%\tProcessing images {i} from total of {img_count}", end="")

        x = Image.open(left_images[i]).resize((IMG_HEIGHT, IMG_WIDTH))
        y = Image.open(right_images[i]).resize((IMG_HEIGHT, IMG_WIDTH))

        x = np.asarray(x)
        x = np.expand_dims(x, 0)

        y = np.asarray(y)
        y = np.expand_dims(y, 0)

        stereo_pairs['left'].append(x)
        stereo_pairs['right'].append(y)

    return np.vstack(stereo_pairs['left']), np.vstack(stereo_pairs['right'])


if __name__ == '__main__':
    # save_data()
    get_file()
