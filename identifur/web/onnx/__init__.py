from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
import uvicorn
import datetime
import io
import logging
import argparse
from PIL import Image
from ...runtime import onnx


INPUT_SIZE = (224, 224)


def main():
    logging.basicConfig(level=logging.INFO)

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model-path", default="identifur.onnx.pb")
    argparser.add_argument("--tags-path", default="tags")
    argparser.add_argument("--host", default="127.0.0.1")
    argparser.add_argument("--port", default=3621, type=int)
    args = argparser.parse_args()

    predictor = onnx.Predictor(
        args.model_path, args.tags_path, providers=["CPUExecutionProvider"]
    )

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
        img = Image.open(io.BytesIO(await file.read()))

        start_time = datetime.datetime.now()
        [(predicted_tags, predicted_rating)] = predictor.predict([img])
        dur = datetime.datetime.now() - start_time

        predicted_tags = sorted(
            enumerate(predicted_tags), key=lambda kv: kv[1], reverse=True
        )

        return {
            "tags": [
                {"name": predictor.tags[id], "score": score.item()}
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
