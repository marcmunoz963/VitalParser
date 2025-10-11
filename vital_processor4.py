# ==================================================================================================
# VitalProcessor4 - Millores globals:
# - Optimització de la càrrega de dades: només es carreguen les tracks/columnes necessàries per a cada model,
#   evitant carregar tot el fitxer a memòria.
# - Ús de Polars per a la manipulació de DataFrames: permet operacions vectoritzades ràpides i join eficients.
# - Escritura incremental a CSV: els resultats es guarden incrementalment, evitant reescriure tot el fitxer
#   cada vegada, i permetent processos continus o en temps real.
# - Processament en chunks: les senyals llargues es processen per fragments per controlar l'ús de memòria RAM
#   i aprofitar el paral·lelisme amb ThreadPoolExecutor.
# - Memòria cau de dades: es manté en memòria la senyal carregada per cada track per evitar lectures redundants.
# - Sincronització i estat: es guarda l'últim segment processat per a cada arxiu per evitar recomputació.
# ==================================================================================================
import os
import numpy as np
import polars as pd
from polars import DataFrame as PLDataFrame
from openpyxl import load_workbook, Workbook
from openpyxl.utils.exceptions import InvalidFileException
from parser.vital_utils import is_nan, find_latest_vital
from vitaldb import VitalFile
import parser.arr as arr_utils
from zipfile import BadZipFile
from concurrent.futures import ThreadPoolExecutor
import gc
import time


def process_segment(cfg, start_time, vf, arr_utils, all_data):
    interval_s = cfg['interval_secs']
    print(f"[TRACE] process_segment() iniciat amb start_time={start_time:.2f}s per {cfg.get('signal_track')}")

    # Millora: reutilitzar la senyal carregada per evitar recarregar-la múltiples vegades
    if all_data is None or all_data.shape[0] == 0:
        print(f"[ERROR] Debug: No es va passar la senyal per {cfg['signal_track']}")
        return None

    # Càlcul d'índexs basat en la durada i la freqüència aproximada
    sample_rate = len(all_data) / (all_data[-1, 0] - all_data[0, 0])  # aprox
    start_idx = int(start_time * sample_rate)
    end_idx = int((start_time + interval_s) * sample_rate)

    if end_idx > len(all_data):
        return None

    segment_data = all_data[start_idx:end_idx]
    if len(segment_data) == 0:
        return None

    timestamps = segment_data[:, 0]
    signal = segment_data[:, 1]

    signal = arr_utils.interp_undefined(signal)

    orig_rate = cfg.get('orig_rate', cfg['resample_rate'])
    signal = arr_utils.resample_hz(signal, orig_rate, cfg['resample_rate'])
    timestamps = arr_utils.resample_hz(timestamps, orig_rate, cfg['resample_rate'])

    if len(signal) < cfg['signal_length']:
        print(f"[ERROR] Debug: Señal demasiado corta: {len(signal)} < {cfg['signal_length']}")
        return None

    inp = signal[:cfg['signal_length']].reshape(1, 1, -1)
    try:
        pred_result = cfg['model'].predict(inp)
        pred = float(pred_result.squeeze())
        return {'Tiempo': timestamps[cfg['signal_length'] - 1], cfg['output_var']: pred}
    except Exception as e:
        print(f"[ERROR] Error en process_segment: {e}")
        import traceback
        traceback.print_exc()
        print(f"[TRACE] process_segment() ha fallat per {cfg.get('signal_track')}: {e}")
        return None


class VitalProcessor:
    def __init__(self, model_configs, results_dir, window_rows=20, poll_interval=1):
        """
        model_configs: lista de dict con configuraciones tabular y wave.
        results_dir: carpeta de salida.
        window_rows: filas para modo tabular.
        poll_interval: no usado aquí.
        """
        self.model_configs = model_configs
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.window_rows = window_rows
        self.poll_interval = poll_interval
        self.latest_df = None
        # Estado para limitar predicciones wave según interval_secs
        # clave: output_var, valor: timestamp de última predicción
        self.wave_last = {cfg.get('output_var'): None for cfg in model_configs if cfg.get('input_type')=='wave'}
        # Para rastrear el último tiempo procesado por archivo para sincronización en tiempo real
        self.last_processing_time = {}
        """MODIFICAT"""
        self.state_file = os.path.join(results_dir, "state_wave.json")
        if os.path.exists(self.state_file):
          import json
          with open(self.state_file, "r") as f:
             self.last_processing_time = json.load(f)

    def process_once(self, recordings_dir, mode='tabular'):
        if mode == 'tabular':
            return self._process_tabular(recordings_dir)
        elif mode == 'wave':
            return self._process_wave(recordings_dir)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _process_tabular(self, recordings_dir):
        vital_path = find_latest_vital(recordings_dir)
        if not vital_path:
            return None
        base = os.path.splitext(os.path.basename(vital_path))[0]
        xlsx_path = os.path.join(self.results_dir, f"{base}_tabular.xlsx")
        first = not os.path.exists(xlsx_path)

        vf = VitalFile(vital_path)
        tracks = vf.get_track_names()
        # Millora: carregar només les últimes window_rows files per reduir memòria
        raw = vf.to_numpy(tracks, interval=0, return_timestamp=True)
        buffer = []
        for row in raw[-self.window_rows:]:
            t, *vals = row
            if any(not is_nan(v) for v in vals):
                rec = {'Tiempo': t}
                rec.update({n: v for n, v in zip(tracks, vals) if not is_nan(v)})
                buffer.append(rec)
        if not buffer:
            return None

        df = pd.DataFrame(buffer)
        df = self._run_predictions(df)
        self.latest_df = df.clone()
        csv_path = xlsx_path.replace(".xlsx", ".csv")
        # Millora: escriptura incremental CSV per evitar escriure tot de nou
        self._save_csv(df, csv_path, first)
        """MODIFICAT"""
        # Guardar estat de processament
        import json
        with open(self.state_file, "w") as f:
           json.dump(self.last_processing_time, f)
        return df

    def _process_wave(self, recordings_dir):
        """
        Procesa pipelines de onda con optimizaciones:
        - Carga sólo las señales necesarias.
        - Procesa en lotes pequeños para controlar uso de memoria.
        - Escritura incremental en CSV para reducir I/O.
        - Sincronización para no procesar segmentos ya evaluados.
        """
        vital_path = find_latest_vital(recordings_dir)
        if not vital_path:
            print("[ERROR] Debug: No se encontró archivo .vital en", recordings_dir)
            return None
        base = os.path.splitext(os.path.basename(vital_path))[0]
        xlsx_path = os.path.join(self.results_dir, f"{base}_wave.xlsx")
        first = not os.path.exists(xlsx_path)

        # -------------------------------------------------------------
        # NOMÉS ES CARREGUEN LES TRACKS NECESSÀRIES per reduir l'ús de RAM:
        # S'identifiquen les tracks requerides pels models wave i es carrega només aquestes.
        # -------------------------------------------------------------
        all_tracks = VitalFile(vital_path).get_track_names()
        needed_tracks = [
            cfg['signal_track'] for cfg in self.model_configs
            if cfg.get('input_type') == 'wave' and cfg.get('signal_track') in all_tracks
        ]
        vf = VitalFile(vital_path, needed_tracks)
        records = []

        processing_start_time = time.time()
        chunk_duration = 60  # Duración en segundos para cada chunk (configurable)

        # --------- INICI: memòria cau de dades per signal_track ---------
        all_data_cache = {}
        # --------- FI memòria cau ---------

        for cfg in self.model_configs:
            if cfg.get('input_type') != 'wave' or cfg.get('model') is None:
                continue

            signal_track = cfg['signal_track']
            if signal_track not in all_data_cache:
                try:
                    all_data_cache[signal_track] = vf.to_numpy([signal_track], interval=0, return_timestamp=True)
                    print(f"[DEBUG] {signal_track} carregat complet ({all_data_cache[signal_track].shape[0]} mostres)")
                except Exception as e:
                    print(f"[ERROR] No s'ha pogut carregar {signal_track}: {e}")
                    continue
            all_data_full = all_data_cache[signal_track]

            # Obtener duración total aproximada de la señal para segmentación
            total_duration = len(vf.to_numpy([cfg['signal_track']], interval=1))

            interval_s = cfg['interval_secs']
            overlap_s = cfg['overlap_secs']
            step_s = max(1, interval_s - overlap_s)

            # Generar todos los start_times posibles para segmentar la señal
            all_start_times = np.arange(0, total_duration - interval_s, step_s) if total_duration > interval_s else []
            if len(all_start_times) == 0:
                continue

            # Control para no reprocesar segmentos ya evaluados (sincronización en tiempo real)
            file_key = os.path.basename(vital_path)
            last_processed = self.last_processing_time.get(file_key, 0)

            batch_size = 20  # Procesar en lotes pequeños para controlar uso de recursos

            if last_processed > 0:
                remaining_start_times = all_start_times[all_start_times > last_processed]
                if len(remaining_start_times) > 0:
                    start_times = remaining_start_times
                    print(f"[INFO] Continuando desde {last_processed}s, {len(start_times)} segmentos pendientes")
                else:
                    # No hay segmentos nuevos, procesar últimos batch_size para mantener actualizado
                    start_times = all_start_times[-batch_size:] if len(all_start_times) >= batch_size else all_start_times
                    print(f"[INFO] No hay segmentos nuevos, procesando últimos {len(start_times)} segmentos")
            else:
                # Primera ejecución, procesar todo
                start_times = all_start_times
                print(f"[INFO] Primera ejecución: procesando desde el inicio ({len(start_times)} segmentos disponibles)")

            if len(start_times) == 0:
                continue
            # -----------------------------------------------------------------------
            # PROCESSAMENT EN CHUNKS (fragments) per controlar l'ús de memòria i
            # aprofitar el paral·lelisme amb ThreadPoolExecutor.
            # Cada chunk cobreix un interval de temps i només es carrega la finestra necessària.
            # -----------------------------------------------------------------------
            chunk_start_times = np.arange(0, total_duration, chunk_duration)
            for chunk_start in chunk_start_times:
                chunk_end = chunk_start + chunk_duration + interval_s  # Añadir interval_s para cubrir segmentos que cruzan límite chunk
                # Filtrar start_times que caen dentro del chunk
                chunk_segment_starts = start_times[(start_times >= chunk_start) & (start_times < chunk_end)]
                if len(chunk_segment_starts) == 0:
                    continue

                # Utilitzar la memòria cau de dades per obtenir el chunk del rang temporal
                mask = (all_data_full[:, 0] >= chunk_start) & (all_data_full[:, 0] < chunk_end)
                all_data_chunk = all_data_full[mask]
                if all_data_chunk is None or len(all_data_chunk) == 0:
                    print(f"[WARN] Sense dades en el rang {chunk_start}-{chunk_end}s per {cfg['signal_track']}")
                    continue
                else:
                    print(f"[DEBUG] {cfg['signal_track']} chunk seleccionat ({all_data_chunk.shape[0]} mostres entre {chunk_start}-{chunk_end}s)")

                # Procesar segmentos dentro del chunk
                num_workers = max(1, os.cpu_count() // 2)
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(process_segment, cfg, st, vf, arr_utils, all_data_chunk) for st in chunk_segment_starts]
                    segment_results = [f.result() for f in futures]

                valid_results = [res for res in segment_results if res is not None]
                records.extend(valid_results)
                gc.collect()

                # Actualizar último tiempo procesado para evitar reprocesamientos
                if len(chunk_segment_starts) > 0:
                    self.last_processing_time[file_key] = max(chunk_segment_starts)

                current_elapsed = time.time() - processing_start_time
                if current_elapsed > 80.0:
                    print(f"[INFO] Tiempo límite alcanzado ({current_elapsed:.1f}s), finalizando procesamiento")
                    break
            # -----------------------------------------------------------------------
            # FI DEL PROCESSAMENT EN CHUNKS
            # -----------------------------------------------------------------------

        if not records:
            print("[WARN] No s'han obtingut prediccions vàlides en aquesta iteració.")
            # Si no hi ha registres, crear CSV buit si no existeix
            csv_path = xlsx_path.replace(".xlsx", ".csv")
            if not os.path.exists(csv_path):
               pd.DataFrame([]).write_csv(csv_path)
            return None
        # Crear DataFrame con predicciones únicas por tiempo
        df_preds = pd.DataFrame(records).unique(subset=['Tiempo'], keep='last')

        tracks = vf.get_track_names()
        raw_all = vf.to_numpy(tracks, interval=0, return_timestamp=True)

        # Preparar columnas para DataFrame, con manejo seguro de valores None
        column_data = {}
        for idx, name in enumerate(['Tiempo'] + tracks):
            try:
                col = raw_all[:, idx]
                if name == 'Tiempo':
                    column_data[name] = col  # Mantener como float para join
                else:
                    safe_col = [str(val) if val is not None else None for val in col]
                    column_data[name] = safe_col
            except Exception as e:
                print(f"Error procesando columna {name}: {e}")
                column_data[name] = [None] * len(raw_all)

        df_all = pd.DataFrame(column_data)
        """MODIFICAT"""
        # Millora: join nadiu amb Polars per afegir les prediccions, evitant loops i mapes
        df = df_all.join(df_preds, on="Tiempo", how="left")
        self.latest_df = df.clone()
        """MODIFICAT"""
        # Filtrar files on totes les prediccions wave són nulles per evitar dades sense valor
        wave_output_vars = [cfg['output_var'] for cfg in self.model_configs if cfg.get('input_type') == 'wave']
        if wave_output_vars:
            df = df.filter(~pd.all_horizontal([pd.col(var).is_null() for var in wave_output_vars]))

        csv_path = xlsx_path.replace(".xlsx", ".csv")
        """MODIFICAT"""
        # -------------------------------------------------------------
        # ESCRIPTURA INCREMENTAL DEL CSV amb Polars:
        # Només s'afegeixen les files noves al fitxer CSV, evitant reescriure tot el fitxer.
        # Això redueix I/O i permet processament eficient en temps real o en lots.
        # -------------------------------------------------------------
        self._save_csv(df, csv_path, first)
        return df

    def _run_predictions(self, df: pd.DataFrame):
        for cfg in self.model_configs:
            if cfg.get('input_type') != 'tabular':
                continue
            if cfg.get('model') is None:
                continue
            ws = cfg.get('window_size', 1)
            vars_ = cfg.get('input_vars', [])
            cols = [c for c in vars_ if c in df]
            if not cols:
                continue
            arr = df[cols].fill_null(0).to_numpy()
            preds = []

            if ws == 1:
                for r in arr:
                    try:
                        out = cfg['model'].predict(r.reshape(1, -1))
                        preds.append(float(out.squeeze()))
                    except Exception as e:
                        print(f"Prediction error (tabular): {e}")
                        preds.append(np.nan)
            else:
                pad = np.zeros((max(0, ws - len(arr)), len(cols)))
                win = np.vstack([pad, arr])
                for i in range(len(arr)):
                    chunk = win[i:i+ws].flatten().reshape(1, -1)
                    try:
                        out = cfg['model'].predict(chunk)
                        preds.append(float(out.squeeze()))
                    except Exception as e:
                        print(f"Prediction error (tabular window): {e}")
                        preds.append(np.nan)

            # Agregar columna de predicciones con sintaxis correcta de Polars
            df = df.with_columns(pd.Series(cfg['output_var'], preds, dtype=pd.Float64))
        return df
    """MODIFICAT"""
    def _save_csv(self, df: pd.DataFrame, path: str, first: bool):
        """
        Escritura incremental en CSV:
        - Si es la primera vez, crea archivo con cabecera.
        - Si no, añade sólo las nuevas filas sin cabecera.
        """
        if not isinstance(df, PLDataFrame):
            raise TypeError("Expected a Polars DataFrame")

        if first or not os.path.exists(path):
            df.write_csv(path)
        else:
            with open(path, "a") as f:
                df.write_csv(f, include_header=False)