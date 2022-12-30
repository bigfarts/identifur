#!/usr/bin/env python3
import markupsafe
import datetime
import io
import base64
import logging
import argparse
from PIL import Image
import torch
from torchvision import transforms
from identifur import models
import flask


class Predictor:
    def __init__(self, model, labels, input_size, device):
        self.model = model
        self.labels = labels
        self.input_size = input_size
        self.device = device

    def predict(self, image):
        y_pred = self.model(
            transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(self.input_size),
                ]
            )(image.convert("RGB"))
            .unsqueeze(0)
            .to(self.device),
        )
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.detach().cpu()
        y_pred = y_pred[0]

        return {self.labels[i]: p for i, p in enumerate(y_pred)}


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
        weights=None,
        num_labels=len(labels),
        requires_grad=False,
    ).to(device)
    model.eval()
    model.freeze()

    predictor = Predictor(model, labels, input_size, device)

    app = create_app(predictor)
    app.run(host=args.host, port=args.port)


_BASE_TEMPLATE = """
<!doctype html>
<html>
    <head>
        <meta charset="utf-8">
        <title>identifur</title>
    </head>
    <body>
        <form method="POST" enctype="multipart/form-data" action="/">
            <input type="file" name="image" />
            <button type="submit">Predict</button>
        </form>
        {rest}
    </body>
</html>
"""


def create_app(predictor):
    app = flask.Flask(__name__)

    @app.route("/", methods=["GET"])
    def index():
        return _BASE_TEMPLATE.format(rest="")

    @app.route("/", methods=["POST"])
    def predict():
        def _inner():
            img = flask.request.files.get("image")
            if img is None:
                return "<p>No image supplied.</p>"

            try:
                img_data = img.stream.read()
                img = Image.open(io.BytesIO(img_data))

                start_time = datetime.datetime.now()
                predictions = predictor.predict(img)
                dur = datetime.datetime.now() - start_time

                predictions = sorted(
                    ((name, score) for name, score in predictions.items()),
                    key=lambda kv: kv[1],
                    reverse=True,
                )
                table = []
                table.append("<table>")
                for name, score in predictions[:50]:
                    table.append(
                        f"<tr><td>{markupsafe.escape(name)}</td><td>{score * 100:.5f}%</td></tr>"
                    )
                table.append("</table>")
                return f"""<p><img src="data:{img.get_format_mimetype()};base64,{base64.b64encode(img_data).decode('utf-8')}" height="300"></p><p>prediction took {dur.total_seconds() * 100}ms.</p>{"".join(table)}"""
            except Exception as e:
                return f"<p>Error: {markupsafe.escape(e)}</p>"

        return _BASE_TEMPLATE.format(rest=_inner())

    return app


if __name__ == "__main__":
    main()
