import pandas as pd
import numpy as np
import matplotlib.patches as mpatches

from matplotlib import pyplot as plt


def rle_to_mask(rle_string, imsize=(256, 1600), fill_color=(1, 0, 0, 0.5)):
    """
    Converts run-length encoded pairs into an RGBA image.
    """
    mask = np.zeros(shape=imsize + (4,))
    nums = [int(x) for x in rle_string.split()]
    N = len(nums)
    for i in range(N//2):
        n, length = nums[2*i:2*i+2]
        ns = [n - 1 + i for i in range(length)]
        ys, xs = np.unravel_index(ns, shape=imsize, order='F')
        mask[ys, xs, :] = fill_color
    return mask


def rle_to_indices(rle_string, imsize=(256, 1600), fill_color=(1, 0, 0, 0.5)):
    """
    Converts run-length encoded pairs into pairs of indices.
    Returns a tuple of (array of x values, array of y values)
    """
    nums = [int(x) for x in rle_string.split()]
    N = len(nums)
    xs = np.asarray([])
    ys = np.asarray([])
    for i in range(N//2):
        n, length = nums[2*i:2*i+2] # Are the pixels in the dataset 1-indexed, instead of 0-indexed?
        ns = [n - 1 + i for i in range(length)]
        ys_, xs_ = np.unravel_index(ns, shape=imsize, order='F')
        xs = np.append(xs, xs_)
        ys = np.append(ys, ys_)
    return xs.astype(int), ys.astype(int)


def display_img_with_mask(df, image_id, figsize=(16, 12), display_classes=[], x_window=(None, None)):
    display_classes = display_classes or [1, 2, 3, 4]
    fname = f'../data/train_images/{image_id}'
    img = plt.imread(fname)

    plt.figure(figsize=figsize)
    img = img[:, x_window[0]:x_window[1]]
    plt.imshow(img)

    colors = [
        (1, 0, 0, 0.25),
        (0, 1, 0, 0.25),
        (0, 0, 1, 0.25),
    ]

    df_mask = df.ImageId == image_id
    df_mask &= df.ClassId.isin(display_classes)
    handles, labels = [], []
    for color, row in zip(colors, df[df_mask].sort_values(by='ClassId').itertuples()):
        mask = rle_to_mask(row.EncodedPixels, fill_color=color)
        mask = mask[:, x_window[0]:x_window[1]]
        plt.imshow(mask)
        handles.append(mpatches.Patch(color=color, label=f'Defect {row.ClassId}'))
    plt.legend(handles=handles)