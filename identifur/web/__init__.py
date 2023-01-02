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
import torch
from torchvision import transforms, models as tv_models
import torchvision.transforms.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.reshape_transforms import vit_reshape_transform
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from identifur import models
import timm


def _image_without_transparency(img: Image.Image):
    if img.mode == "RGBA":
        img2 = Image.new("RGB", img.size, (0, 0, 0))
        img2.paste(img, mask=img.split()[3])
        return img2

    return img.convert("RGB")


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, "constant")


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

    if isinstance(model, tv_models.ConvNeXt):
        return [model.features[-1]], lambda xs: xs

    if isinstance(model, tv_models.EfficientNet):
        return [model.features[-1]], lambda xs: xs

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
        tags = [line.rstrip("\n") for line in f]
    tag_ids = {label: i for i, label in enumerate(tags)}

    (model, input_size) = models.MODELS[args.base_model]
    model = models.LitModel.load_from_checkpoint(
        args.checkpoint,
        model=model,
        pretrained=False,
        num_labels=len(tags) + 3,
        requires_grad=True,
    ).to(device)
    model.eval()
    # model.freeze()

    image_transforms = transforms.Compose(
        [
            SquarePad(),
            transforms.Resize(input_size),
        ]
    )

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

        input_img = image_transforms(img)
        input_img_tensor = transforms.ToTensor()(input_img).unsqueeze(0).to(device)

        start_time = datetime.datetime.now()
        predictions = model(input_img_tensor).detach().cpu()[0]
        dur = datetime.datetime.now() - start_time

        predicted_tags = torch.sigmoid(predictions[:-3])
        predicted_rating = torch.softmax(predictions[-3:], 0)

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

    @app.post(
        "/gradcam",
        responses={200: {"content": {"image/png": {}}}},
        response_class=Response,
    )
    async def gradcam(
        file: UploadFile,
        tag: str = Form(...),
    ):
        try:
            tag_id = tag_ids[tag]
        except KeyError:
            raise HTTPException(status_code=400, detail="unknown tag")

        img = _image_without_transparency(Image.open(io.BytesIO(await file.read())))

        input_img = image_transforms(img)
        input_img_tensor = transforms.ToTensor()(input_img).unsqueeze(0).to(device)

        grayscale_cam = (
            np.asarray(
                transforms.ToPILImage()(
                    cam(
                        input_tensor=input_img_tensor,
                        targets=[ClassifierOutputTarget(tag_id)],
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
