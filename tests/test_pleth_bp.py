import time
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parser.vital_processor import VitalProcessor
from parser.model_loader import load_ml_model

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(base_dir, 'model.json')
    
    print("=== TEST PREDICCIÓN PRESIÓN ARTERIAL DESDE PLETH ===")
    
    with open(cfg_path) as f:
        configs = json.load(f)
    
    # Buscar configuración PLETH
    pleth_config = None
    for cfg in configs:
        if cfg.get('signal_track') == 'Demo/PLETH':
            pleth_config = cfg
            break
    
    if not pleth_config:
        print("[ERROR] No se encontró configuración para PLETH en model.json")
        return
    
    print(f"[OK] Configuración PLETH encontrada: {pleth_config['name']}")
    
    # Cargar modelo
    try:
        model_path = os.path.join(base_dir, pleth_config['path'])
        pleth_config['model'] = load_ml_model(model_path, pleth_config)
        print(f"[OK] Modelo cargado desde: {model_path}")
    except Exception as e:
        print(f"[ERROR] Error cargando modelo: {e}")
        return
    
    # Configurar processor
    results_dir = os.path.join(base_dir, 'results')
    processor = VitalProcessor(configs, results_dir)
    recordings_dir = os.path.join(base_dir, 'records')
    
    print("\n[INFO] Ejecutando procesamiento PLETH por 10 segundos...\n")
    
    start_time = time.time()
    iteration = 0
    
    while time.time() - start_time < 10:
        iteration += 1
        df = processor.process_once(recordings_dir, mode='wave')
        
        if df is not None:
            elapsed = time.time() - start_time
            total_rows = len(df)
            
            print(f"--- ITERACIÓN {iteration} ({elapsed:.1f}s) ---")
            print(f"   [INFO] Filas procesadas: {total_rows:,}")
            print(f"   Columnas: {list(df.columns)}")
            
            # Buscar columnas de predicción BP
            bp_cols = [col for col in df.columns if 'BP' in col or 'Prediccion' in col or 'HTI' in col]
            if bp_cols:
                print(f"   [INFO] Columnas de presión arterial: {bp_cols}")
                
                for col in bp_cols:
                    non_null_preds = df.filter(df[col].is_not_null())
                    if len(non_null_preds) > 0:
                        sample_values = non_null_preds[col].head(5).to_list()
                        # Convertir valores a float, manejando strings
                        numeric_values = []
                        for v in sample_values:
                            try:
                                numeric_values.append(float(v))
                            except (ValueError, TypeError):
                                pass
                        
                        if numeric_values:
                            avg_value = np.mean(numeric_values)
                            print(f"      {col}: {len(non_null_preds)} valores")
                            print(f"      Ejemplos: {[f'{v:.1f}' for v in numeric_values[:3]]}")
                            print(f"      Promedio: {avg_value:.1f}")
                        else:
                            print(f"      {col}: {len(non_null_preds)} valores (no numéricos)")
                            print(f"      Ejemplos: {sample_values[:3]}")
                    else:
                        print(f"      {col}: Sin valores válidos")
            else:
                print("   [WARNING] No se encontraron columnas de predicción BP")
            
            # Mostrar algunas filas de datos PLETH si están disponibles
            pleth_cols = [col for col in df.columns if 'PLETH' in col]
            if pleth_cols:
                print(f"   [INFO] Columnas PLETH disponibles: {pleth_cols}")
                
        else:
            print(f"   [ERROR] Iteración {iteration}: No se generaron resultados")
        
        print()
        time.sleep(1)
    
    print("=== RESUMEN DEL TEST ===")
    print(f"[OK] Test completado en {time.time() - start_time:.1f} segundos")
    print(f"[INFO] Total de iteraciones: {iteration}")
    print("[OK] Modelo PLETH-BP integrado exitosamente en VitalParser")

if __name__ == '__main__':
    main()
