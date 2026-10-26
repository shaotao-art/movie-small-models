import pickle
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import json
import os


def read_image(image_path: str) -> np.ndarray:
    if not image_path or not isinstance(image_path, str):
        raise ValueError("Invalid image path provided.")
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(img: np.ndarray, save_path: str):
    dir_name = os.path.dirname(save_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img)


def get_image_base64(image_np: np.ndarray, format: str = "PNG"):
    pil_image = Image.fromarray(image_np)
    img_byte_arr = BytesIO()
    pil_image.save(img_byte_arr, format=format)
    img_byte_arr = img_byte_arr.getvalue()
    encoded_image = base64.b64encode(img_byte_arr)
    encoded_image_text = encoded_image.decode("utf-8")
    base64_code = f"data:image/{format.lower()};base64,{encoded_image_text}"
    return base64_code


def load_from_pickle(input_file):
    with open(input_file, "rb") as f:
        return pickle.load(f)


def save_to_pickle(data, output_file):
    with open(output_file, "wb") as f:
        pickle.dump(data, f)


def load_text(input_file, encoding="utf-8"):
    with open(input_file, "r", encoding=encoding) as f:
        return f.read()


def save_text(text, output_file):
    if os.path.dirname(output_file) and not os.path.exists(
        os.path.dirname(output_file)
    ):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write(text)


def load_json(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
