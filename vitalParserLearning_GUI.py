import os
import json
import threading
import tkinter as tk
from parser.model_loader import load_ml_model
from parser.vital_processor import VitalProcessor
from parser.gui import VitalApp

# Carga la configuración sin cargar los modelos aún
def load_configs(base_dir):
    cfg_path = os.path.join(base_dir, 'model.json')
    with open(cfg_path) as f:
        configs = json.load(f)
    for cfg in configs:
        cfg['model'] = None  # Lazy loading posterior
    return configs

# Carga de modelos en segundo plano
def load_models_async(configs, base_dir, callback=None):
    def task():
        for cfg in configs:
            try:
                model_path = os.path.join(base_dir, cfg['path'])
                cfg['model'] = load_ml_model(model_path, cfg)
            except Exception as e:
                print(f"[ERROR] No se pudo cargar el modelo {cfg['path']}: {e}")
        if callback:
            callback()
    threading.Thread(target=task, daemon=True).start()

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    configs = load_configs(base_dir)
    results_dir = os.path.join(base_dir, 'results')
    
    root = tk.Tk()
    processor = VitalProcessor(configs, results_dir)
    app = VitalApp(root, processor, configs)

    # Cargar modelos en segundo plano después de iniciar la GUI
    load_models_async(configs, base_dir)

    root.mainloop()
