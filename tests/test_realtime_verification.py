import time
import json
import os
import sys
import numpy as np
import polars as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parser.vital_processor import VitalProcessor
from parser.model_loader import load_ml_model
from parser.vital_utils import find_latest_vital

def check_realtime_processing():
    """Verifica si el procesamiento es en tiempo real"""
    print("=== VERIFICACIÓN DE PROCESAMIENTO EN TIEMPO REAL ===")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(base_dir, 'model.json')
    
    with open(cfg_path) as f:
        configs = json.load(f)
    
    # Cargar modelo PLETH
    pleth_config = None
    for cfg in configs:
        if cfg.get('signal_track') == 'Demo/PLETH':
            pleth_config = cfg
            model_path = os.path.join(base_dir, cfg['path'])
            cfg['model'] = load_ml_model(model_path, cfg)
            print(f"[OK] Modelo PLETH cargado: {cfg['name']}")
            break
    
    if not pleth_config:
        print("[ERROR] No se encontró configuración PLETH")
        return
    
    # Configurar processor
    results_dir = os.path.join(base_dir, 'results')
    processor = VitalProcessor(configs, results_dir)
    recordings_dir = os.path.join(base_dir, 'records')
    
    print(f"\n[INFO] Ejecutando procesamiento en tiempo real por 10 segundos...")
    print("=" * 60)
    
    start_time = time.time()
    iteration = 0
    
    while time.time() - start_time < 10:
        iteration += 1
        iteration_start = time.time()
        
        df = processor.process_once(recordings_dir, mode='wave')
        iteration_time = time.time() - iteration_start
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
            
            print(f"[INFO] {elapsed_total:.1f}s | Iter: {iteration} | "
                  f"Filas: {total_rows:,} | "
                  f"Pred: {prediction_count} | "
                  f"Tiempo: {iteration_time:.2f}s | "
                  f"RT: {'[OK]' if iteration_time < 1 else '[WARNING]'}")
            
            # Verificar si es tiempo real
            if iteration_time > 1:
                print(f"   [WARNING] Procesamiento lento: {iteration_time:.2f}s > 1s")
        else:
            print(f"[INFO] {elapsed_total:.1f}s | Iter: {iteration} | [ERROR] Sin resultados")
        
        time.sleep(0.5)  # Pausa para no saturar
    
    print("=" * 60)
    print("[OK] Verificación de tiempo real completada")

def check_prediction_column():
    """Verifica si se genera la columna de predicción"""
    print("\n=== VERIFICACIÓN DE COLUMNA DE PREDICCIÓN ===")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(base_dir, 'model.json')
    
    with open(cfg_path) as f:
        configs = json.load(f)
    
    # Cargar modelo PLETH
    pleth_config = None
    for cfg in configs:
        if cfg.get('signal_track') == 'Demo/PLETH':
            pleth_config = cfg
            model_path = os.path.join(base_dir, cfg['path'])
            cfg['model'] = load_ml_model(model_path, cfg)
            break
    
    if not pleth_config:
        print("[ERROR] No se encontró configuración PLETH")
        return
    
    # Configurar processor
    results_dir = os.path.join(base_dir, 'results')
    processor = VitalProcessor(configs, results_dir)
    recordings_dir = os.path.join(base_dir, 'records')
    
    print("[INFO] Ejecutando una iteración para verificar columnas...")
    
    df = processor.process_once(recordings_dir, mode='wave')
    
    if df is not None:
        print(f"DataFrame generado con {len(df)} filas")
        print(f"Columnas disponibles: {list(df.columns)}")
        
        # Buscar columnas de predicción
        bp_cols = [col for col in df.columns if 'BP_Prediction' in col]
        pred_cols = [col for col in df.columns if 'Prediccion' in col or 'HTI' in col]
        
        print(f"\nColumnas de predicción encontradas:")
        print(f"   BP_Prediction: {bp_cols}")
        print(f"   Otras predicciones: {pred_cols}")
        
        if bp_cols:
            col = bp_cols[0]
            non_null_preds = df.filter(df[col].is_not_null())
            print(f"\nAnálisis de {col}:")
            print(f"   Total valores: {len(df)}")
            print(f"   Valores no nulos: {len(non_null_preds)}")
            
            if len(non_null_preds) > 0:
                values = []
                for v in non_null_preds[col].head(10).to_list():
                    try:
                        val = float(v)
                        if not np.isnan(val):
                            values.append(val)
                    except (ValueError, TypeError):
                        pass
                
                if values:
                    print(f"   Rango: {min(values):.1f} - {max(values):.1f} mmHg")
                    print(f"   Promedio: {np.mean(values):.1f} mmHg")
                    print(f"   Ejemplos: {[f'{v:.1f}' for v in values[:5]]}")
                else:
                    print(f"    Sin valores numéricos válidos")
        else:
            print("No se encontró columna BP_Prediction")
    else:
        print("No se generó DataFrame")

def check_csv_output():
    """Verifica si se genera archivo CSV de resultados"""
    print("\n=== VERIFICACIÓN DE ARCHIVOS DE RESULTADO ===")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'results')
    
    print(f"Directorio de resultados: {results_dir}")
    
    if not os.path.exists(results_dir):
        print("Directorio de resultados no existe")
        return
    
    # Buscar archivos de resultado
    result_files = []
    for file in os.listdir(results_dir):
        if file.endswith(('.xlsx', '.csv')):
            result_files.append(file)
    
    print(f" Archivos de resultado encontrados: {len(result_files)}")
    
    if result_files:
        print("\n Archivos disponibles:")
        for file in sorted(result_files):
            file_path = os.path.join(results_dir, file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            print(f"    {file}")
            print(f"      Tamaño: {file_size:.1f} KB")
            print(f"      Modificado: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Verificar si contiene predicciones BP
            if 'BP_Prediction' in file or 'wave' in file:
                print(f"       Contiene predicciones PLETH-BP")
            
            # Mostrar formato
            if file.endswith('.csv'):
                print(f"      Formato: CSV")
            elif file.endswith('.xlsx'):
                print(f"      Formato: Excel")
    else:
        print("No se encontraron archivos de resultado")
    
    # Verificar si VitalParser genera CSV
    print(f"\n Verificando configuración de salida...")
    
    # Buscar en el código si se genera CSV
    try:
        with open(os.path.join(base_dir, 'parser', 'vital_processor.py'), 'r') as f:
            content = f.read()
            if 'csv' in content.lower():
                print(" El código incluye funcionalidad CSV")
            else:
                print("  El código no incluye funcionalidad CSV explícita")
    except:
        print("  No se pudo verificar el código")

def main():
    print("[INFO] VERIFICACIÓN COMPLETA DEL MODELO PLETH-BP")
    print("=" * 60)
    print(f"[INFO] Fecha/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Verificar procesamiento en tiempo real
    check_realtime_processing()
    
    # 2. Verificar columna de predicción
    check_prediction_column()
    
    # 3. Verificar archivos de resultado
    check_csv_output()
    
    print("\n" + "=" * 60)
    print(" Verificación completa finalizada")

if __name__ == '__main__':
    main()
