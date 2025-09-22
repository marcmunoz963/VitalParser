import os
import sys
import joblib
import numpy as np
import torch
from torch import nn
from tensorflow.keras.models import load_model as keras_load_model
import importlib.util

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


class PythonModelWrapper:
    """
    Wraps a Python module with a run() function to provide a .predict() method.
    """
    def __init__(self, module, cfg):
        self.module = module
        self.cfg = cfg

    def predict(self, X):
        """
        Adapts the run() function interface to predict() interface
        """
        try:
            # X should be a 1D signal array
            signal_data = X.flatten() if hasattr(X, 'flatten') else np.array(X).flatten()
            
            # Create input structure expected by run() function
            inp = {
                self.cfg.get('signal_track', 'Demo/PLETH'): {
                    'vals': signal_data,
                    'srate': self.cfg.get('resample_rate', 100)
                }
            }
            
            result = self.module.run(inp, {}, self.cfg)
            
            if result and len(result) > 0 and len(result[0]) > 0:
                # Extract prediction values
                predictions = []
                for pred in result[0]:
                    if isinstance(pred, dict) and 'val' in pred:
                        predictions.append(pred['val'])
                
                if len(predictions) == 1:
                    return np.array([predictions[0]])
                else:
                    return np.array(predictions)
            return np.array([0])  # Default if no prediction
            
        except Exception as e:
            print(f"Error en PythonModelWrapper.predict: {e}")
            import traceback
            traceback.print_exc()
            return np.array([0])


def load_ml_model(path, cfg=None):
    """
    Load a ML model from path. Supports:
      - .joblib: sklearn or pickled PyTorch
      - .h5: Keras
      - .pt/.pth: PyTorch
      - .py: Python module with run() function
    Returns an object with .predict().
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    model_dir = os.path.dirname(path)
    if model_dir and model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    ext = os.path.splitext(path)[1].lower()
    
    # PYTHON MODULE: load .py file as module
    if ext == '.py':
        spec = importlib.util.spec_from_file_location("custom_model", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load Python module from {path}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if not hasattr(module, 'run'):
            raise TypeError("Python module must have a run() function")
        
        return PythonModelWrapper(module, cfg or {})
    
    # JOBLIB: could be sklearn or torch
    elif ext == '.joblib':
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