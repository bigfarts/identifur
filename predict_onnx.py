#!/usr/bin/env python3
import numpy as np
import argparse
from PIL import Image
import onnxruntime as ort
from torchvision import transforms


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()


def _image_without_transparency(img: Image.Image):
    if img.mode == "RGBA":
        img2 = Image.new("RGB", img.size, (0, 0, 0))
        img2.paste(img, mask=img.split()[3])
        return img2

    return img.convert("RGB")


INPUT_SIZE = 224


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("image")
    argparser.add_argument("--model-path", default="identifur.onnx.pb")
    argparser.add_argument("--tags-path", default="tags")
    argparser.add_argument("--top", default=50, type=int)
    args = argparser.parse_args()

    with open(args.tags_path, "rt", encoding="utf-8") as f:
        tags = [line.rstrip("\n") for line in f]

    orig = _image_without_transparency(Image.open(args.image))
    max_side = orig.width if orig.width > orig.height else orig.height
    orig = orig.resize(
        (orig.width * INPUT_SIZE // max_side, orig.height * INPUT_SIZE // max_side)
    )

    x_off = (INPUT_SIZE - orig.width) // 2
    y_off = (INPUT_SIZE - orig.height) // 2

    img = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (0, 0, 0))
    img.paste(orig, (x_off, y_off))
    img.save("test.png")

    img = np.array([(np.asarray(img, np.float32) / 255.0).transpose((2, 0, 1))])

    ort_sess = ort.InferenceSession(args.model_path, providers=["CPUExecutionProvider"])

    predictions = ort_sess.run(None, {"images": img})[0][0]

    predicted_tags = sigmoid(predictions[:-3])
    predicted_rating = softmax(predictions[-3:])

    predicted_tags = sorted(
        enumerate(predicted_tags), key=lambda kv: kv[1], reverse=True
    )

    for id, score in predicted_tags[: args.top]:
        print(f"{tags[id]}\t{score}")

    print("")

    for name, p in zip(["safe", "questionable", "explicit"], predicted_rating):
        print(f"{name}\t{p}")


if __name__ == "__main__":
    main()
