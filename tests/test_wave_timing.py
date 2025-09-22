import time
import json
import os
import sys

# Agregar el directorio padre al path para importar módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parser.vital_processor import VitalProcessor
from parser.model_loader import load_ml_model

def main():
    print("=== ANÁLISIS DE TIMING DEL PROCESAMIENTO WAVE ===\n")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(base_dir, 'model.json')
    with open(cfg_path) as f:
        configs = json.load(f)
    
    results_dir = os.path.join(base_dir, 'results')
    processor = VitalProcessor(configs, results_dir)
    recordings_dir = os.path.join(base_dir, 'records')
    
    # Cargar modelos
    print("1. Cargando modelos...")
    model_start = time.time()
    for cfg in configs:
        if 'path' in cfg:
            model_path = os.path.join(base_dir, cfg['path'])
            cfg['model'] = load_ml_model(model_path)
    model_time = time.time() - model_start
    print(f"   [OK] Modelos cargados en {model_time:.2f}s\n")
    
    print("2. Ejecutando procesamiento wave con timing detallado...")
    
    # Ejecutar 3 iteraciones para ver el patrón
    for i in range(3):
        print(f"\n--- ITERACIÓN {i+1} ---")
        iteration_start = time.time()
        
        df = processor.process_once(recordings_dir, mode='wave')
        
        iteration_time = time.time() - iteration_start
        
        if df is not None:
            # Análisis de datos
            total_rows = len(df)
            pred_cols = [col for col in df.columns if col not in ['Tiempo'] and 'Demo/' not in col and col != 'EVENT']
            
            if pred_cols:
                non_null_preds = df.filter(df[pred_cols[0]].is_not_null())
                predictions_count = len(non_null_preds)
                
                # Calcular tiempo de datos procesados
                if 'Tiempo' in df.columns:
                    tiempo_col = df['Tiempo']
                    # Convertir a float si es necesario
                    try:
                        tiempo_min = float(tiempo_col.min())
                        tiempo_max = float(tiempo_col.max())
                        duracion_datos = tiempo_max - tiempo_min
                    except:
                        # Si hay problemas con tipos, usar valores alternativos
                        tiempo_values = [float(x) for x in tiempo_col.to_list() if x is not None]
                        if tiempo_values:
                            tiempo_min = min(tiempo_values)
                            tiempo_max = max(tiempo_values)
                            duracion_datos = tiempo_max - tiempo_min
                        else:
                            duracion_datos = 0
                    
                    print(f"   [INFO] Datos procesados:")
                    print(f"      - Total filas: {total_rows:,}")
                    print(f"      - Predicciones: {predictions_count:,}")
                    print(f"      - Duración de datos: {duracion_datos:.1f}s")
                    print(f"   [INFO] Tiempo de procesamiento: {iteration_time:.2f}s")
                    
                    # Calcular ratio tiempo real vs procesamiento
                    ratio = iteration_time / duracion_datos if duracion_datos > 0 else 0
                    print(f"   [INFO] Ratio procesamiento/datos: {ratio:.2f}x")
                    
                    if ratio < 1:
                        print(f"   [OK] TIEMPO REAL: Procesa más rápido que los datos")
                    elif ratio < 2:
                        print(f"   [WARNING] CERCA DE TIEMPO REAL: Ligero retraso")
                    else:
                        print(f"   [ERROR] NO TIEMPO REAL: Retraso significativo")
                        
                    # Throughput
                    throughput = total_rows / iteration_time
                    print(f"   [INFO] Throughput: {throughput:.0f} filas/segundo")
                    
        else:
            print(f"   [ERROR] No se generaron resultados en {iteration_time:.2f}s")
        
        # Esperar un poco entre iteraciones
        if i < 2:
            time.sleep(2)
    
    print(f"\n=== ANÁLISIS DE LA SECUENCIA DE PROCESAMIENTO ===")
    print("Basado en el código, la secuencia es:")
    print("1. [INFO] find_latest_vital() - Busca el archivo .vital más reciente")
    print("2. [INFO] VitalFile() - Carga el archivo vital")
    print("3. [INFO] Itera por cada modelo wave en model_configs")
    print("4. [INFO] Calcula duración total de la señal")
    print("5. [INFO] Genera start_times con solapamiento")
    print("6. [INFO] ThreadPoolExecutor procesa segmentos en paralelo")
    print("7. [INFO] Cada segmento: extrae datos -> interpola -> remuestrea -> predice")
    print("8. [INFO] Combina predicciones con datos originales")
    print("9. [INFO] Guarda en Excel")
    print("10. [OK] Retorna DataFrame completo")

if __name__ == '__main__':
    main()
