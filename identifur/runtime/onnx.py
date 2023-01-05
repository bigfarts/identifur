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
        providers=ort.get_available_providers(),
        provider_options=None,
    ):
        self.ort_sess = ort.InferenceSession(
            model_path, providers=providers, provider_options=provider_options
        )
        self.tags = self.ort_sess.get_modelmeta().custom_metadata_map["tags"].split(" ")

    def predict(self, images):
        predictions = self.ort_sess.run(
            None, {"images": np.stack([preprocess_image(image) for image in images])}
        )[0]

        return [
            (sigmoid(prediction[:-3]), softmax(prediction[-3:]))
            for prediction in predictions
        ]
