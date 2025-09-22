import time
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parser.vital_processor import VitalProcessor
from parser.model_loader import load_ml_model

def test_batch_processing():
    """Test para verificar el procesamiento por lotes"""
    print("=== TEST DE PROCESAMIENTO POR LOTES ===")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(base_dir, 'model.json')
    
    with open(cfg_path) as f:
        configs = json.load(f)
    
    # Cargar solo el modelo PLETH
    pleth_config = None
    for cfg in configs:
        if cfg.get('signal_track') == 'Demo/PLETH':
            pleth_config = cfg
            model_path = os.path.join(base_dir, cfg['path'])
            cfg['model'] = load_ml_model(model_path, cfg)
            print(f"[OK] Modelo PLETH cargado: {cfg['name']}")
            print(f"   Signal length: {cfg['signal_length']}")
            print(f"   [INFO] Interval: {cfg['interval_secs']}s")
            print(f"   [INFO] Overlap: {cfg['overlap_secs']}s")
            break
    
    if not pleth_config:
        print("[ERROR] No se encontró configuración PLETH")
        return
    
    # Configurar processor
    results_dir = os.path.join(base_dir, 'results')
    processor = VitalProcessor(configs, results_dir)
    recordings_dir = os.path.join(base_dir, 'records')
    
    print(f"\n[INFO] Ejecutando test de procesamiento por lotes...")
    print("=" * 60)
    
    # Ejecutar varias iteraciones para ver el comportamiento por lotes
    for iteration in range(1, 2):
        print(f"\n--- ITERACIÓN {iteration} ---")
        iteration_start = time.time()
        
        df = processor.process_once(recordings_dir, mode='wave')
        iteration_time = time.time() - iteration_start
        
        if df is not None:
            total_rows = len(df)
            
            # Verificar predicciones
            bp_cols = [col for col in df.columns if 'BP_Prediction' in col]
            if bp_cols:
                non_null_preds = df.filter(df[bp_cols[0]].is_not_null())
                prediction_count = len(non_null_preds)
            else:
                prediction_count = 0
            
            print(f"[INFO] Tiempo de procesamiento: {iteration_time:.2f}s")
            print(f"[INFO] Filas procesadas: {total_rows:,}")
            print(f"[INFO] Predicciones generadas: {prediction_count}")
            
            # Mostrar estado de procesamiento
            if hasattr(processor, 'last_processing_time'):
                for file_key, last_time in processor.last_processing_time.items():
                    print(f"[INFO] {file_key}: último procesado en {last_time:.1f}s")
            
            # Mostrar algunas predicciones si existen
            if prediction_count > 0:
                values = []
                for v in non_null_preds[bp_cols[0]].head(5).to_list():
                    try:
                        val = float(v)
                        if not np.isnan(val):
                            values.append(val)
                    except (ValueError, TypeError):
                        pass
                
                if values:
                    print(f"[INFO] Ejemplos de predicciones: {[f'{v:.1f}' for v in values]}")
        else:
            print("[ERROR] No se generaron resultados")
        
        # Pausa entre iteraciones
        if iteration < 3:
            print(" Esperando 2 segundos antes de la siguiente iteración...")
            time.sleep(2)
    
    print("=" * 60)
    print("[OK] Test de procesamiento por lotes completado")

def main():
    print("[INFO] VERIFICACIÓN DE PROCESAMIENTO POR LOTES")
    print("=" * 60)
    print(f"[INFO] Fecha/Hora: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_batch_processing()

if __name__ == '__main__':
    main()
