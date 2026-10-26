import math
import cv2
import numpy as np


def smart_resize(
    height: int,
    width: int,
    max_pixels: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
):

    if height < factor or width < factor:
        raise ValueError(
            f"height:{height} or width:{width} must be larger than factor:{factor}"
        )
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def resize_image_short_side(image_array: np.ndarray, x: int):
    height, width = image_array.shape[:2]

    if width <= height:
        new_width = x
        new_height = int(height * (x / width))
    else:
        new_width = int(width * (x / height))
        new_height = x

    img_resized = cv2.resize(
        image_array, (new_width, new_height), interpolation=cv2.INTER_AREA
    )
    return img_resized


def resize_image_max_side(image_array: np.ndarray, x: int):
    height, width = image_array.shape[:2]

    if width <= height:
        new_height = x
        new_width = int(width * (x / height))
    else:
        new_width = x
        new_height = int(height * (x / width))

    img_resized = cv2.resize(
        image_array, (new_width, new_height), interpolation=cv2.INTER_AREA
    )
    return img_resized


def smart_resize_img(image_array: np.ndarray, max_pixels: int):
    height, width = image_array.shape[:2]
    new_height, new_width = smart_resize(height, width, max_pixels)
    img_resized = cv2.resize(
        image_array, (new_width, new_height), interpolation=cv2.INTER_AREA
    )
    return img_resized
