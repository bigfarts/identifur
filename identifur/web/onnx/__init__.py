from fastapi import FastAPI, UploadFile, Form, Response, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
import uvicorn
import functools
import numpy as np
import datetime
import io
import logging
import argparse
from PIL import Image
import onnxruntime as ort


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()


INPUT_SIZE = 224


def _image_without_transparency(img: Image.Image):
    if img.mode == "RGBA":
        img2 = Image.new("RGB", img.size, (0, 0, 0))
        img2.paste(img, mask=img.split()[3])
        return img2

    return img.convert("RGB")


def main():
    logging.basicConfig(level=logging.INFO)

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model-path", default="identifur.onnx.pb")
    argparser.add_argument("--tags-path", default="tags")
    argparser.add_argument("--host", default="127.0.0.1")
    argparser.add_argument("--port", default=3621, type=int)
    args = argparser.parse_args()

    with open(args.tags_path, "rt", encoding="utf-8") as f:
        tags = [line.rstrip("\n") for line in f]

    ort_sess = ort.InferenceSession(args.model_path, providers=["CPUExecutionProvider"])

    app = FastAPI()
    app.mount(
        "/static",
        StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")),
        name="static",
    )

    @app.get("/")
    async def index():
        return FileResponse(
            os.path.join(os.path.dirname(__file__), "index.html"),
            media_type="text/html",
        )

    @app.post("/predict")
    async def predict(
        file: UploadFile,
        top: int = Form(50),
    ):
        orig = _image_without_transparency(Image.open(io.BytesIO(await file.read())))

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

        start_time = datetime.datetime.now()
        predictions = ort_sess.run(None, {"images": img})[0][0]
        dur = datetime.datetime.now() - start_time

        predicted_tags = sigmoid(predictions[:-3])
        predicted_rating = softmax(predictions[-3:])

        predicted_tags = sorted(
            enumerate(predicted_tags), key=lambda kv: kv[1], reverse=True
        )

        return {
            "tags": [
                {"name": tags[id], "score": score.item()}
                for id, score in predicted_tags[:top]
            ],
            "rating": {
                name: p.item()
                for name, p in zip(
                    ["safe", "questionable", "explicit"], predicted_rating
                )
            },
            "elapsed_secs": dur.total_seconds(),
        }

    uvicorn.run(app, host=args.host, port=args.port)
