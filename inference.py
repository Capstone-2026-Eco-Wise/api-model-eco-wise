"""
inference.py — EcoWise Waste Classifier
========================================
Standalone inference module.
Diserahkan ke Backend bersama model_ecowise_production.keras & class_names.json

Usage:
    from inference import EcoWiseInference
    engine = EcoWiseInference("model_ecowise_production.keras", "class_names.json")
    result = engine.predict("path/to/image.jpg")
"""

import os
import json
import numpy as np
from PIL import Image
import io

import tensorflow as tf

# Pastikan custom objects terdaftar saat load model
# (diperlukan karena model menggunakan ChannelAttentionLayer & FocalLoss)


class ChannelAttentionLayer(tf.keras.layers.Layer):
    """
    Custom Layer: Channel Attention (Squeeze-and-Excitation).
    Harus terdaftar saat load model .keras.
    """
    def __init__(self, reduction_ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.fc1 = tf.keras.layers.Dense(
            max(channels // self.reduction_ratio, 8),
            activation="relu", use_bias=False)
        self.fc2 = tf.keras.layers.Dense(
            channels, activation="sigmoid", use_bias=False)
        super().build(input_shape)

    def call(self, inputs, training=None):
        attention = self.fc1(inputs)
        attention = self.fc2(attention)
        return inputs * attention

    def get_config(self):
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config


class FocalLoss(tf.keras.losses.Loss):
    """Custom Loss: Focal Loss."""
    def __init__(self, gamma=2.0, alpha=0.25, num_classes=3, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes

    def call(self, y_true, y_pred):
        y_true_int = tf.cast(y_true, tf.int32)
        y_true_oh  = tf.one_hot(
            tf.squeeze(y_true_int, axis=-1) if tf.rank(y_true_int) > 1 else y_true_int,
            depth=self.num_classes)
        y_pred     = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce         = -y_true_oh * tf.math.log(y_pred)
        weight     = self.alpha * y_true_oh * tf.pow(1.0 - y_pred, self.gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha,
                        "num_classes": self.num_classes})
        return config


# ── Fallback class names jika class_names.json tidak ada ────────────────────
DEFAULT_CLASS_NAMES = ["Anorganik", "B3", "Organik"]
IMG_SIZE = (224, 224)


class EcoWiseInference:
    """
    Production inference engine untuk EcoWise Waste Classifier.

    Parameters
    ----------
    model_path       : path ke file .keras
    class_names_path : path ke class_names.json (opsional)
    """

    def __init__(self, model_path: str, class_names_path: str = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "ChannelAttentionLayer": ChannelAttentionLayer,
                "FocalLoss": FocalLoss,
            },
        )

        if class_names_path and os.path.exists(class_names_path):
            with open(class_names_path) as f:
                data = json.load(f)
            self.class_names = data.get("class_names", DEFAULT_CLASS_NAMES)
        else:
            self.class_names = DEFAULT_CLASS_NAMES

        print(f"✅ Model loaded. Kelas: {self.class_names}")

    # ── Core preprocessing ───────────────────────────────────────────────────
    @staticmethod
    def _preprocess(img_array: np.ndarray) -> np.ndarray:
        """
        Preprocessing wajib sebelum inference.
        Sama dengan pipeline training: MobileNetV2 preprocess_input → range [-1, 1]
        """
        img_array = img_array.astype("float32")
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        return img_array

    # ── Predict dari file path ───────────────────────────────────────────────
    def predict(self, img_path: str) -> dict:
        """
        Prediksi dari path file gambar.

        Returns
        -------
        {
            "label": str,
            "confidence": float (0-1),
            "all_scores": {"Anorganik": float, "B3": float, "Organik": float}
        }
        """
        img       = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
        arr       = np.array(img)
        arr       = self._preprocess(arr)
        arr       = np.expand_dims(arr, axis=0)           # (1, 224, 224, 3)
        return self._run_inference(arr)

    # ── Predict dari bytes (untuk REST API upload) ────────────────────────────
    def predict_from_bytes(self, img_bytes: bytes) -> dict:
        """
        Prediksi dari raw bytes (cocok untuk endpoint FastAPI UploadFile).

        Parameters
        ----------
        img_bytes : bytes hasil await file.read()
        """
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize(IMG_SIZE)
        arr = np.array(img)
        arr = self._preprocess(arr)
        arr = np.expand_dims(arr, axis=0)
        return self._run_inference(arr)

    # ── Predict dari numpy array ─────────────────────────────────────────────
    def predict_from_array(self, img_array: np.ndarray) -> dict:
        """
        Prediksi dari numpy array (shape: H x W x 3, dtype uint8 atau float32).
        """
        img = Image.fromarray(img_array.astype("uint8")).resize(IMG_SIZE)
        arr = np.array(img, dtype="float32")
        arr = self._preprocess(arr)
        arr = np.expand_dims(arr, axis=0)
        return self._run_inference(arr)

    # ── Internal inference ───────────────────────────────────────────────────
    def _run_inference(self, batch: np.ndarray) -> dict:
        preds    = self.model.predict(batch, verbose=0)[0]   # (num_classes,)
        pred_idx = int(np.argmax(preds))
        return {
            "label":      self.class_names[pred_idx],
            "confidence": float(np.max(preds)),
            "all_scores": {
                name: float(score)
                for name, score in zip(self.class_names, preds)
            },
        }


# ── CLI Quick Test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python inference.py <model.keras> <image_path>")
        print("       python inference.py model_ecowise_production.keras test.jpg")
        sys.exit(1)

    engine = EcoWiseInference(
        model_path=sys.argv[1],
        class_names_path="class_names.json" if os.path.exists("class_names.json") else None,
    )
    result = engine.predict(sys.argv[2])

    print(f"\n📦 Hasil Prediksi")
    print(f"   Label      : {result['label']}")
    print(f"   Confidence : {result['confidence'] * 100:.2f}%")
    print(f"   Semua skor : {result['all_scores']}")
