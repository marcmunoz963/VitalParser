import time
import json
import os
import sys

# Agregar el directorio padre al path para importar módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parser.vital_processor import VitalProcessor
from parser.model_loader import load_ml_model

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Subir un nivel desde tests/
    cfg_path = os.path.join(base_dir, 'model.json')
    with open(cfg_path) as f:
        configs = json.load(f)
    
    results_dir = os.path.join(base_dir, 'results')
    processor = VitalProcessor(configs, results_dir)
    
    recordings_dir = os.path.join(base_dir, 'records')
    
    start_time = time.time()
    for cfg in configs:
        model_path = os.path.join(base_dir, cfg['path'])
        cfg['model'] = load_ml_model(model_path)
    
    while time.time() - start_time < 20:
        df = processor.process_once(recordings_dir, mode='tabular')
        if df is not None:
            print(f"Tiempo: {time.time() - start_time:.2f}s | Shape: {df.shape}")
            print(f"  Columnas: {list(df.columns)}")
            # Buscar columnas de predicción (no son tracks de Demo/)
            pred_cols = [col for col in df.columns if col not in ['Tiempo'] and 'Demo/' not in col and col != 'EVENT']
            if pred_cols:
                print(f"  Columnas de predicción encontradas: {pred_cols}")
                for col in pred_cols:
                    non_null_values = df.filter(df[col].is_not_null())
                    if len(non_null_values) > 0:
                        sample_values = non_null_values[col].head(3).to_list()
                        print(f"    {col}: {len(non_null_values)} valores, ejemplos: {sample_values}")
            else:
                print("  No se encontraron columnas de predicción")
        time.sleep(1)

if __name__ == '__main__':
    main()
