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
from vitaldb import VitalFile

def analyze_vital_files(records_dir):
    """Analiza todos los archivos .vital disponibles"""
    print("=== ANÁLISIS DE ARCHIVOS .VITAL ===")
    
    vital_files = []
    for root, dirs, files in os.walk(records_dir):
        for file in files:
            if file.endswith('.vital'):
                vital_path = os.path.join(root, file)
                vital_files.append(vital_path)
    
    print(f"[INFO] Total de archivos .vital encontrados: {len(vital_files)}")
    
    for i, vital_path in enumerate(vital_files):
        try:
            vf = VitalFile(vital_path)
            tracks = vf.get_track_names()
            
            # Buscar tracks relevantes
            pleth_tracks = [t for t in tracks if 'PLETH' in t]
            art_tracks = [t for t in tracks if 'ART' in t]
            
            file_size = os.path.getsize(vital_path) / (1024*1024)  # MB
            
            print(f"\n Archivo {i+1}: {os.path.basename(vital_path)}")
            print(f"   [INFO] Tamaño: {file_size:.1f} MB")
            print(f"   [INFO] Total tracks: {len(tracks)}")
            print(f"   [INFO] Tracks PLETH: {pleth_tracks}")
            print(f"    Tracks ART: {art_tracks}")
            
            if pleth_tracks:
                # Analizar duración de datos PLETH
                try:
                    pleth_data = vf.to_numpy([pleth_tracks[0]], interval=1)
                    if pleth_data is not None and len(pleth_data) > 0:
                        duration = len(pleth_data)  # segundos aproximados
                        print(f"   [INFO] Duración PLETH: ~{duration} segundos")
                        print(f"   [INFO] Muestras PLETH: {len(pleth_data):,}")
                        
                        # Verificar calidad de señal - manejar diferentes formatos
                        if pleth_data.ndim == 2 and pleth_data.shape[1] >= 2:
                            signal_values = pleth_data[:, 1]  # columna de valores
                        elif pleth_data.ndim == 1:
                            signal_values = pleth_data  # array 1D
                        else:
                            signal_values = pleth_data.flatten()  # convertir a 1D
                            
                        non_nan_ratio = np.sum(~np.isnan(signal_values)) / len(signal_values)
                        if non_nan_ratio > 0:
                            signal_range = np.nanmax(signal_values) - np.nanmin(signal_values)
                            print(f"   [OK] Calidad señal: {non_nan_ratio*100:.1f}% válida, rango: {signal_range:.2f}")
                        else:
                            print(f"   [ERROR] Señal sin datos válidos")
                    else:
                        print(f"   [WARNING] Sin datos PLETH disponibles")
                except Exception as e:
                    print(f"   [ERROR] Error analizando PLETH: {e}")
            
        except Exception as e:
            print(f"   [ERROR] Error analizando {vital_path}: {e}")
    
    return vital_files

def test_pleth_bp_performance(records_dir, duration_seconds=60):
    """Test completo de desempeño del modelo PLETH-BP"""
    print(f"\n=== TEST DE DESEMPEÑO PLETH-BP ({duration_seconds}s) ===")
    
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
    
    # Métricas de desempeño
    metrics = {
        'iterations': 0,
        'total_predictions': 0,
        'prediction_values': [],
        'processing_times': [],
        'total_rows_processed': 0,
        'errors': 0
    }
    
    print(f"\n[INFO] Ejecutando test de desempeño por {duration_seconds} segundos...")
    print("=" * 60)
    
    start_time = time.time()
    
    while time.time() - start_time < duration_seconds:
        iteration_start = time.time()
        metrics['iterations'] += 1
        
        try:
            df = processor.process_once(records_dir, mode='wave')
            iteration_time = time.time() - iteration_start
            metrics['processing_times'].append(iteration_time)
            
            if df is not None:
                metrics['total_rows_processed'] += len(df)
                
                # Analizar predicciones BP
                bp_cols = [col for col in df.columns if 'BP_Prediction' in col]
                
                for col in bp_cols:
                    non_null_preds = df.filter(df[col].is_not_null())
                    if len(non_null_preds) > 0:
                        values = []
                        for v in non_null_preds[col].to_list():
                            try:
                                val = float(v)
                                if not np.isnan(val) and val > 0:
                                    values.append(val)
                            except (ValueError, TypeError):
                                pass
                        
                        metrics['total_predictions'] += len(values)
                        metrics['prediction_values'].extend(values)
                
                elapsed = time.time() - start_time
                
                # Mostrar progreso cada 10 segundos
                if elapsed % 10 < 1 and elapsed > 10:
                    avg_time = np.mean(metrics['processing_times'])
                    throughput = metrics['total_rows_processed'] / elapsed
                    pred_rate = metrics['total_predictions'] / elapsed
                    
                    print(f"[INFO] {elapsed:.0f}s | Iter: {metrics['iterations']} | "
                          f"Filas: {metrics['total_rows_processed']:,} | "
                          f"Pred: {metrics['total_predictions']} | "
                          f"Throughput: {throughput:.0f} filas/s | "
                          f"Pred/s: {pred_rate:.2f}")
            
            else:
                metrics['errors'] += 1
                
        except Exception as e:
            metrics['errors'] += 1
            print(f"[ERROR] Error en iteración {metrics['iterations']}: {e}")
        
        # Pequeña pausa para no saturar
        time.sleep(0.1)
    
    # Análisis final
    total_time = time.time() - start_time
    print("=" * 60)
    print("[INFO] RESULTADOS DEL TEST DE DESEMPEÑO")
    print("=" * 60)
    
    print(f"[INFO] Tiempo total: {total_time:.1f}s")
    print(f"[INFO] Iteraciones: {metrics['iterations']}")
    print(f"[INFO] Filas procesadas: {metrics['total_rows_processed']:,}")
    print(f"[INFO] Predicciones generadas: {metrics['total_predictions']}")
    print(f"[ERROR] Errores: {metrics['errors']}")
    
    if metrics['processing_times']:
        avg_time = np.mean(metrics['processing_times'])
        min_time = np.min(metrics['processing_times'])
        max_time = np.max(metrics['processing_times'])
        
        print(f"\n RENDIMIENTO:")
        print(f"   Tiempo promedio por iteración: {avg_time:.2f}s")
        print(f"   Tiempo mínimo: {min_time:.2f}s")
        print(f"   Tiempo máximo: {max_time:.2f}s")
        print(f"   Throughput: {metrics['total_rows_processed']/total_time:.0f} filas/s")
        print(f"   Tasa de predicciones: {metrics['total_predictions']/total_time:.2f} pred/s")
    
    if metrics['prediction_values']:
        values = np.array(metrics['prediction_values'])
        
        print(f"\n[INFO] ANÁLISIS DE PREDICCIONES:")
        print(f"   Total predicciones válidas: {len(values)}")
        print(f"   Rango: {np.min(values):.1f} - {np.max(values):.1f} mmHg")
        print(f"   Promedio: {np.mean(values):.1f} mmHg")
        print(f"   Mediana: {np.median(values):.1f} mmHg")
        print(f"   Desviación estándar: {np.std(values):.1f} mmHg")
        
        # Análisis de distribución
        normal_range = np.sum((values >= 90) & (values <= 140))
        high_range = np.sum(values > 140)
        low_range = np.sum(values < 90)
        
        print(f"\n[INFO] DISTRIBUCIÓN FISIOLÓGICA:")
        print(f"   Normal (90-140 mmHg): {normal_range} ({normal_range/len(values)*100:.1f}%)")
        print(f"   Alta (>140 mmHg): {high_range} ({high_range/len(values)*100:.1f}%)")
        print(f"   Baja (<90 mmHg): {low_range} ({low_range/len(values)*100:.1f}%)")
        
        # Ejemplos de predicciones
        sample_size = min(10, len(values))
        sample_indices = np.random.choice(len(values), sample_size, replace=False)
        sample_values = values[sample_indices]
        print(f"   Ejemplos: {[f'{v:.1f}' for v in sample_values]}")
    
    print(f"\n[OK] Test de desempeño completado")
    return metrics

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    records_dir = os.path.join(base_dir, 'records')
    
    print("[INFO] EVALUACIÓN DE DESEMPEÑO - PREDICCIÓN PLETH-BP")
    print("=" * 60)
    print(f"[INFO] Directorio de datos: {records_dir}")
    print(f"[INFO] Fecha/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Analizar archivos disponibles
    vital_files = analyze_vital_files(records_dir)
    
    if not vital_files:
        print("[ERROR] No se encontraron archivos .vital")
        return
    
    # 2. Test de desempeño
    metrics = test_pleth_bp_performance(records_dir, duration_seconds=30)
    
    # 3. Recomendaciones
    print(f"\ RECOMENDACIONES:")
    if metrics['total_predictions'] > 0:
        print("   [OK] El modelo está funcionando correctamente")
        if np.mean(metrics['processing_times']) < 5:
            print("   [OK] Rendimiento adecuado para tiempo real")
        else:
            print("   [WARNING] Considerar optimización para mejor rendimiento")
    else:
        print("   [ERROR] El modelo no está generando predicciones")
        print("   Revisar configuración y datos de entrada")

if __name__ == '__main__':
    main()


