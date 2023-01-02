from PIL import Image
import numpy as np


def _image_without_transparency(img: Image.Image):
    if img.mode == "RGBA":
        img2 = Image.new("RGB", img.size, (0, 0, 0))
        img2.paste(img, mask=img.split()[3])
        return img2

    return img.convert("RGB")


def preprocess_image(img, input_size=(224, 224)):
    orig = _image_without_transparency(img)

    iw, ih = input_size

    max_side = orig.width if orig.width > orig.height else orig.height
    orig = orig.resize((orig.width * iw // max_side, orig.height * ih // max_side))

    x_off = (iw - orig.width) // 2
    y_off = (ih - orig.height) // 2

    img = Image.new("RGB", (iw, ih), (0, 0, 0))
    img.paste(orig, (x_off, y_off))

    return (np.asarray(img, np.float32) / 255.0).transpose((2, 0, 1))
