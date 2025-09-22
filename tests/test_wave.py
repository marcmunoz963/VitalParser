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
    
    # Cargar modelos
    for cfg in configs:
        if 'path' in cfg:
            model_path = os.path.join(base_dir, cfg['path'])
            cfg['model'] = load_ml_model(model_path)
    
    start_time = time.time()
    while time.time() - start_time < 5:  # Solo 5 segundos para prueba rápida
        df = processor.process_once(recordings_dir, mode='wave')
        if df is not None:
            print(f"Tiempo: {time.time() - start_time:.2f}s | Shape: {df.shape}")
            print(f"  Columnas: {list(df.columns)}")
            # Mostrar algunas predicciones si existen
            pred_cols = [col for col in df.columns if col not in ['Tiempo'] and 'Demo/' not in col and col != 'EVENT']
            if pred_cols:
                print(f"  Columnas de predicción encontradas: {pred_cols}")
                for col in pred_cols:
                    non_null_preds = df.filter(df[col].is_not_null())
                    if len(non_null_preds) > 0:
                        sample_values = non_null_preds[col].head(3).to_list()
                        print(f"    {col}: {len(non_null_preds)} valores, ejemplos: {sample_values}")
            else:
                print("  No se encontraron columnas de predicción")
        else:
            print("  No se generaron resultados")
        time.sleep(1)

if __name__ == '__main__':
    main()
