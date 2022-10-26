import os

import numpy as np
import pandas as pd
import torch
import torch.utils
import torchvision
import logging


log = logging.getLogger()


class LabeledImageWindowDataset(torch.utils.data.Dataset):
    def __init__(self, window_df: pd.DataFrame, image_dir, device=None, read_mode=None):
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f'Using device: {device}')

        if read_mode and read_mode.lower() == 'gray':
            self.read_mode = torchvision.io.ImageReadMode.GRAY
        else:
            self.read_mode = torchvision.io.ImageReadMode.UNCHANGED
        log.info(f'Read mode is {self.read_mode}')

        self.window_df = window_df
        self.image_dir = image_dir

        test_row = self.window_df.iloc[0, :]
        test_img = torchvision.io.read_image(os.path.join(self.image_dir, test_row.ImageId), self.read_mode)
        hw = test_row.window_size // 2
        extra = test_row.window_size % 2
        x_min, x_max = int(test_row.instance_center_x - hw), int(test_row.instance_center_x + hw + extra)
        test_img = test_img[..., x_min:x_max]

        log.info(f'Image shapes: {test_img.shape}')
        n_channels = test_img.shape[0]
        means = stds = (0.5,) * n_channels
        embiggened_size = tuple(int(s * 1.12) for s in test_img.shape[1:])
        log.info(f'Embiggened size: {embiggened_size}')

        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(embiggened_size, torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.RandomCrop(test_img.shape[1:]),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ConvertImageDtype(torch.float32),
            torchvision.transforms.Normalize(mean=means, std=stds),
        ])
        self.device = device

    def __getitem__(self, n):
        row = self.window_df.iloc[n, :]
        img = torchvision.io.read_image(os.path.join(self.image_dir, row.ImageId), self.read_mode)
        hw = row.window_size // 2
        extra = row.window_size % 2
        x_min, x_max = int(row.instance_center_x - hw), int(row.instance_center_x + hw + extra)
        img = img[..., x_min:x_max]
        return self.preprocess(img), row.ClassId

    def __len__(self):
        return self.window_df.shape[0]


class ImageWindowDataset(torch.utils.data.Dataset):
    def __init__(self, window_df: pd.DataFrame, image_dir, defect_classes=None, num_samples=None, random_state=42, device=None, read_mode=None):
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f'Using device: {device}')

        if defect_classes:
            log.info(f'Using only defect classes {defect_classes}')
            mask = window_df.ClassId.isin(defect_classes)
            window_df = window_df[mask]

        if num_samples:
            log.info(f'Taking {num_samples} samples')
            window_df = window_df.sample(n=num_samples, random_state=random_state)

        if read_mode and read_mode.lower() == 'gray':
            self.read_mode = torchvision.io.ImageReadMode.GRAY
        else:
            self.read_mode = torchvision.io.ImageReadMode.UNCHANGED
        log.info(f'Read mode is {self.read_mode}')

        self.window_df = window_df
        self.image_dir = image_dir

        test_row = self.window_df.iloc[0, :]
        test_img = torchvision.io.read_image(os.path.join(self.image_dir, test_row.ImageId), self.read_mode)
        hw = test_row.window_size // 2
        extra = test_row.window_size % 2
        x_min, x_max = int(test_row.instance_center_x - hw), int(test_row.instance_center_x + hw + extra)
        test_img = test_img[..., x_min:x_max]

        log.info(f'Image shapes: {test_img.shape}')
        n_channels = test_img.shape[0]
        means = stds = (0.5,) * n_channels
        embiggened_size = tuple(int(s * 1.12) for s in test_img.shape[1:])
        log.info(f'Embiggened size: {embiggened_size}')

        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(embiggened_size, torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.RandomCrop(test_img.shape[1:]),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ConvertImageDtype(torch.float32),
            torchvision.transforms.Normalize(mean=means, std=stds),
        ])
        self.device = device

    def __getitem__(self, n):
        row = self.window_df.iloc[n, :]
        img = torchvision.io.read_image(os.path.join(self.image_dir, row.ImageId), self.read_mode)
        hw = row.window_size // 2
        extra = row.window_size % 2
        x_min, x_max = int(row.instance_center_x - hw), int(row.instance_center_x + hw + extra)
        img = img[..., x_min:x_max]
        return self.preprocess(img)

    def __len__(self):
        return self.window_df.shape[0]


def has_blank_space(img, tol=25):
    """
    The heuristic for detecting blank space is if one of the image columns is close to zero.
    If the mean of all pixel vals is less than tol, we say it's close to zero.

    :param img: A grayscale image with the x-dimension in the first index.
    :param tol: A column's mean pixel values should be below this level.
    :return:
    """
    return np.any([np.mean(col) < tol for col in img])


def denorm_image(img: torch.Tensor):
    """
    De-normalizes an image, so that when displayed it appears like a natural scene instead of looking real glitchy.

    :param img: Normalized image.
    :return: de-normed image
    """
    n_channels = img.shape[0]
    device = img.device
    # These params would move a tanh output in the range [-1, 1] to [0, 1]
    scale = torch.Tensor([0.5] * n_channels).to(device)
    bias = torch.Tensor([0.5] * n_channels).to(device)
    scale = torch.unsqueeze(torch.unsqueeze(scale, 1), 2)
    bias = torch.unsqueeze(torch.unsqueeze(bias, 1), 2)
    with torch.no_grad():
        return img * scale + bias
