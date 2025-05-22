import os
import sys
import joblib
import numpy as np
import torch
from torch import nn
from tensorflow.keras.models import load_model as keras_load_model

class PyTorchWrapper:
    """
    Wraps a PyTorch nn.Module to provide a .predict() method.
    """
    def __init__(self, model: nn.Module):
        self.model = model.eval()

    def predict(self, X):
        arr = np.asarray(X, dtype=np.float32)
        tensor = torch.from_numpy(arr).float()
        with torch.no_grad():
            out = self.model(tensor)
        return out.numpy()


def load_ml_model(path):
    """
    Load a ML model from path. Supports:
      - .joblib: sklearn or pickled PyTorch
      - .h5: Keras
      - .pt/.pth: PyTorch
    Returns an object with .predict().
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    model_dir = os.path.dirname(path)
    if model_dir and model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    ext = os.path.splitext(path)[1].lower()
    # JOBLIB: could be sklearn or torch
    if ext == '.joblib':
        raw = joblib.load(path)
        if isinstance(raw, nn.Module):
            return PyTorchWrapper(raw)
        if hasattr(raw, 'predict'):
            return raw
        raise TypeError("Loaded joblib object has no .predict() and is not a PyTorch model")
    # KERAS
    elif ext == '.h5':
        model = keras_load_model(path)
        if not hasattr(model, 'predict'):
            raise TypeError("Loaded Keras model lacks .predict()")
        return model
    # PYTORCH
    elif ext in ('.pt', '.pth'):
        raw = torch.load(path, map_location='cpu')
        if not isinstance(raw, nn.Module):
            raise TypeError("Loaded file is not a PyTorch nn.Module")
        return PyTorchWrapper(raw)
    else:
        raise ValueError(f"Unsupported model extension: {ext}")