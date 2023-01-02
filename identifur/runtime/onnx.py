import numpy as np
import onnxruntime as ort
from . import preprocess_image


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x, axis=None):
    return np.exp(x - np.max(x, axis)) / np.exp(x - np.max(x, axis)).sum(axis)


class Predictor:
    def __init__(
        self,
        model_path,
        tags_path,
        providers=ort.get_available_providers(),
        provider_options=None,
    ):
        with open(tags_path, "rt", encoding="utf-8") as f:
            self.tags = [line.rstrip("\n") for line in f]

        self.ort_sess = ort.InferenceSession(
            model_path, providers=providers, provider_options=provider_options
        )

    def predict(self, images):
        predictions = self.ort_sess.run(
            None, {"images": np.stack([preprocess_image(image) for image in images])}
        )[0]

        return [
            (sigmoid(prediction[:-3]), softmax(prediction[-3:]))
            for prediction in predictions
        ]
