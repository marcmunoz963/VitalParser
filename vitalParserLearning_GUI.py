import os
import json
import threading
import tkinter as tk
from parser.model_loader import load_ml_model
from parser.vital_processor import VitalProcessor
from parser.gui import VitalApp

# Carga configuración de modelos
def load_configs(base_dir):
    cfg_path = os.path.join(base_dir, 'model.json')
    with open(cfg_path) as f:
        configs = json.load(f)
    for cfg in configs:
        cfg['model'] = load_ml_model(os.path.join(base_dir, cfg['path']))
    return configs

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    configs = load_configs(base_dir)
    results_dir = os.path.join(base_dir, 'results')
    processor = VitalProcessor(configs, results_dir)
    root = tk.Tk()
    app = VitalApp(root, processor, configs)
    root.mainloop()