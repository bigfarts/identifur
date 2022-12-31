#!/usr/bin/env python3
import traceback
import collections
import functools
import numpy as np
import markupsafe
import datetime
import io
import base64
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
import flask
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
            width=model.patch_embed.img_size[0] // model.patch_embed.patch_size[0],
            height=model.patch_embed.img_size[1] // model.patch_embed.patch_size[1],
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
    labels += ["rating: s", "rating: q", "rating: e"]

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

    app = create_app(device, predictor, input_size, cam)
    app.run(host=args.host, port=args.port)


_BASE_TEMPLATE = """
<!doctype html>
<html>
    <head>
        <meta charset="utf-8">
        <title>identifur</title>
    </head>
    <body>
        <form method="POST" enctype="multipart/form-data" action="/" id="form">
            <input type="file" name="image" />
            <input type="hidden" name="image_png_base64" value="" />
            <input type="hidden" name="cam_target_label" value="" />
            <button type="submit">Predict</button>
        </form>
        {rest}
    </body>
</html>
"""


_GRADCAM_SCRIPT = r"""
<script>
    const form = document.getElementById("form");
    const sample = document.getElementById("sample");
    for (const button of document.querySelectorAll("button[data-gradcam-id]")) {
        button.onclick = function () {
            form.elements.cam_target_label.value = this.dataset.gradcamId;
            form.elements.image_png_base64.value = sample.src.replace(/^data:image\/png;base64,/, '');
            form.submit();
        };
    }
</script>
"""


def _image_without_transparency(img: Image.Image):
    if img.mode == "RGBA":
        img2 = Image.new("RGB", img.size, (255, 255, 255))
        img2.paste(img, mask=img.split()[3])
        return img2

    return img.convert("RGB")


def create_app(device, predictor, input_size, cam):
    app = flask.Flask(__name__)

    @app.route("/", methods=["GET"])
    def index():
        return _BASE_TEMPLATE.format(
            rest="",
        )

    @app.route("/", methods=["POST"])
    def predict():
        def _inner():
            try:
                image_png_base64 = flask.request.form.get("image_png_base64", "")
                if image_png_base64:
                    img = Image.open(
                        io.BytesIO(base64.b64decode(image_png_base64)), formats=["PNG"]
                    )
                elif "image" in flask.request.files:
                    img = _image_without_transparency(
                        Image.open(
                            io.BytesIO(flask.request.files["image"].stream.read())
                        )
                    )
                else:
                    return "<p>No image supplied.</p>"

                cam_target_label = flask.request.form.get("cam_target_label", "")
                if cam_target_label:
                    cam_target_label = int(cam_target_label)
                else:
                    cam_target_label = None

                input_img = transforms.Resize(input_size)(img)
                input_img_tensor = (
                    transforms.ToTensor()(input_img).unsqueeze(0).to(device)
                )

                img_buf = io.BytesIO()
                img.save(img_buf, format="PNG")

                if cam_target_label is not None:
                    grayscale_cam = (
                        np.asarray(
                            transforms.ToPILImage()(
                                cam(
                                    input_tensor=input_img_tensor,
                                    targets=[ClassifierOutputTarget(cam_target_label)],
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
                else:
                    img2_buf = img_buf

                start_time = datetime.datetime.now()
                predictions = predictor.predict(input_img_tensor)
                dur = datetime.datetime.now() - start_time

                predictions = sorted(
                    enumerate(predictions), key=lambda kv: kv[1], reverse=True
                )
                table = []
                table.append("<table>")
                for id, score in predictions[:50]:
                    table.append(
                        f"""<tr><td>{markupsafe.escape(predictor.labels[id])} ({id})</td><td>{score * 100:.5f}%</td><td><button type="button" data-gradcam-id="{id}">grad-cam</button></td></tr>"""
                    )
                table.append("</table>")
                return f"""<p>
    <img id="sample" src="data:image/png;base64,{base64.b64encode(img_buf.getvalue()).decode('utf-8')}" height="300">
{f'<img id="gradcam" src="data:image/png;base64,{base64.b64encode(img2_buf.getvalue()).decode("utf-8")}" height="300">' if cam_target_label is not None else ''}
</p>
<p>{f'<strong>showing grad-cam label:</strong> {markupsafe.escape(predictor.labels[cam_target_label])}' if cam_target_label is not None else 'grad-cam off'}</p>
<p>prediction took {dur.total_seconds() * 100}ms.</p>
{"".join(table)}
"""
            except Exception as e:
                return f"<pre>{markupsafe.escape(''.join(traceback.format_exception(e)))}</pre>"

        return _BASE_TEMPLATE.format(
            rest=_inner() + _GRADCAM_SCRIPT,
        )

    return app


if __name__ == "__main__":
    main()
