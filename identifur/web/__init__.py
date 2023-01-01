from fastapi import FastAPI, UploadFile, Form, Response, HTTPException
import multiprocessing
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
import torch
from torchvision import transforms, models as tv_models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.reshape_transforms import vit_reshape_transform
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from identifur import models
import timm


class Predictor:
    def __init__(self, model, labels):
        self.model = model
        self.labels = labels

    def predict(self, image):
        y_pred = self.model(image)
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.detach().cpu()
        return y_pred[0]


def _image_without_transparency(img: Image.Image):
    if img.mode == "RGBA":
        img2 = Image.new("RGB", img.size, (255, 255, 255))
        img2.paste(img, mask=img.split()[3])
        return img2

    return img.convert("RGB")


def _get_gradcam_settings(model):
    if isinstance(model, tv_models.ResNet):
        return [model.layer4[-1]], lambda xs: xs

    if isinstance(model, tv_models.VisionTransformer):
        return [model.encoder.layers[-1].ln_1], functools.partial(
            vit_reshape_transform,
            width=model.image_size // model.patch_size,
            height=model.image_size // model.patch_size,
        )

    if isinstance(model, timm.models.vision_transformer.VisionTransformer):
        return [model.blocks[-1].norm1], functools.partial(
            vit_reshape_transform,
            width=model.patch_embed.img_size[0] // model.patch_embed.patch_size[0],  # type: ignore
            height=model.patch_embed.img_size[1] // model.patch_embed.patch_size[1],  # type: ignore
        )

    raise TypeError(type(model))


def main():
    logging.basicConfig(level=logging.INFO)

    argparser = argparse.ArgumentParser()
    argparser.add_argument("checkpoint")
    argparser.add_argument("--device", default="cuda")
    argparser.add_argument("--base-model", default="vit_l_16")
    argparser.add_argument("--tags-path", default="tags")
    argparser.add_argument("--host", default="127.0.0.1")
    argparser.add_argument("--port", default=3621, type=int)
    args = argparser.parse_args()

    device = torch.device(args.device)

    with open(args.tags_path, "rt", encoding="utf-8") as f:
        labels = [line.rstrip("\n") for line in f]
    labels += ["rating: safe", "rating: questionable", "rating: explicit"]
    label_ids = {label: i for i, label in enumerate(labels)}

    (model, input_size) = models.MODELS[args.base_model]
    model = models.LitModel.load_from_checkpoint(
        args.checkpoint,
        model=model,
        pretrained=False,
        num_labels=len(labels),
        requires_grad=True,
    ).to(device)
    model.eval()
    # model.freeze()

    predictor = Predictor(model, labels)
    target_layers, reshape_transform = _get_gradcam_settings(model.model)
    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform,
        use_cuda=args.device == "cuda",
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
        img = _image_without_transparency(Image.open(io.BytesIO(await file.read())))

        input_img = transforms.Resize(input_size)(img)
        input_img_tensor = transforms.ToTensor()(input_img).unsqueeze(0).to(device)

        start_time = datetime.datetime.now()
        predictions = predictor.predict(input_img_tensor)
        dur = datetime.datetime.now() - start_time

        predictions = sorted(enumerate(predictions), key=lambda kv: kv[1], reverse=True)

        return {
            "predictions": [
                {"label": predictor.labels[id], "score": score.item()}
                for id, score in predictions[:top]
            ],
            "elapsed_secs": dur.total_seconds(),
        }

    @app.post(
        "/gradcam",
        responses={200: {"content": {"image/png": {}}}},
        response_class=Response,
    )
    async def gradcam(
        file: UploadFile,
        label: str = Form(...),
    ):
        try:
            label_id = label_ids[label]
        except KeyError:
            raise HTTPException(status_code=400, detail="unknown label")

        img = _image_without_transparency(Image.open(io.BytesIO(await file.read())))

        input_img = transforms.Resize(input_size)(img)
        input_img_tensor = transforms.ToTensor()(input_img).unsqueeze(0).to(device)

        grayscale_cam = (
            np.asarray(
                transforms.ToPILImage()(
                    cam(
                        input_tensor=input_img_tensor,
                        targets=[ClassifierOutputTarget(label_id)],
                    ).reshape(input_size)
                    * 255
                ).convert("L")
            )
            / 255.0
        )

        img2 = Image.fromarray(
            show_cam_on_image(
                np.asarray(input_img) / 255.0,
                grayscale_cam,
                use_rgb=True,
            )
        )
        img2_buf = io.BytesIO()
        img2.save(img2_buf, format="PNG")
        return Response(img2_buf.getvalue(), media_type="image/png")

    uvicorn.run(app, host=args.host, port=args.port)
