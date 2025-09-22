import time
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parser.vital_processor import VitalProcessor
from parser.model_loader import load_ml_model

def test_realtime_synchronization():
    """Test para verificar que el procesamiento se mantiene sincronizado en tiempo real"""
    print("=== TEST DE SINCRONIZACIÓN EN TIEMPO REAL ===")
    
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
            print(f"   [INFO] Signal length: {cfg['signal_length']}")
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
    
    print(f"\n[INFO] Ejecutando test de sincronización por 15 segundos...")
    print("=" * 60)
    
    start_time = time.time()
    iteration = 0
    processing_times = []
    
    while time.time() - start_time < 15:
        iteration += 1
        iteration_start = time.time()
        
        df = processor.process_once(recordings_dir, mode='wave')
        iteration_time = time.time() - iteration_start
        processing_times.append(iteration_time)
        elapsed_total = time.time() - start_time
        
        if df is not None:
            total_rows = len(df)
            
            # Verificar si se genera la columna de predicción
            bp_cols = [col for col in df.columns if 'BP_Prediction' in col]
            has_predictions = len(bp_cols) > 0
            
            if has_predictions:
                non_null_preds = df.filter(df[bp_cols[0]].is_not_null())
                prediction_count = len(non_null_preds)
            else:
                prediction_count = 0
            
            # Verificar sincronización
            is_realtime = iteration_time < 2.0  # Menos de 2 segundos por iteración
            sync_status = "[OK] RT" if is_realtime else "[WARNING] LENTO"
            
            print(f"[INFO] {elapsed_total:.1f}s | Iter: {iteration} | "
                  f"Filas: {total_rows:,} | "
                  f"Pred: {prediction_count} | "
                  f"Tiempo: {iteration_time:.2f}s | "
                  f"{sync_status}")
            
            # Mostrar estado de sincronización
            if hasattr(processor, 'last_processing_time'):
                for file_key, last_time in processor.last_processing_time.items():
                    print(f"   [INFO] {file_key}: último procesado en {last_time:.1f}s")
        else:
            print(f"[INFO] {elapsed_total:.1f}s | Iter: {iteration} | [ERROR] Sin resultados")
        
        time.sleep(0.5)  # Pausa para no saturar
    
    print("=" * 60)
    print("[INFO] ANÁLISIS DE SINCRONIZACIÓN:")
    
    if processing_times:
        avg_time = np.mean(processing_times)
        min_time = np.min(processing_times)
        max_time = np.max(processing_times)
        
        print(f"   Tiempo promedio: {avg_time:.2f}s")
        print(f"   Tiempo mínimo: {min_time:.2f}s")
        print(f"   Tiempo máximo: {max_time:.2f}s")
        
        # Evaluar sincronización
        realtime_count = sum(1 for t in processing_times if t < 2.0)
        realtime_percentage = (realtime_count / len(processing_times)) * 100
        
        print(f"   Iteraciones en tiempo real: {realtime_count}/{len(processing_times)} ({realtime_percentage:.1f}%)")
        
        if realtime_percentage >= 80:
            print("   [OK] EXCELENTE: Sincronización en tiempo real mantenida")
        elif realtime_percentage >= 60:
            print("   [WARNING] BUENO: Sincronización parcialmente mantenida")
        else:
            print("   [ERROR] MALO: Problemas de sincronización")
    
    print("[OK] Test de sincronización completado")

def main():
    print("[INFO] VERIFICACIÓN DE SINCRONIZACIÓN EN TIEMPO REAL")
    print("=" * 60)
    print(f"[INFO] Fecha/Hora: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_realtime_synchronization()

if __name__ == '__main__':
    main()
