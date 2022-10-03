import numpy as np


def has_blank_space(img, tol=25):
    """
    The heuristic for detecting blank space is if one of the image columns is close to zero.
    If the mean of all pixel vals is less than tol, we say it's close to zero.

    :param img: A grayscale image with the x-dimension in the first index.
    :param tol: A column's mean pixel values should be below this level.
    :return:
    """
    return np.any([np.mean(col) < tol for col in img])