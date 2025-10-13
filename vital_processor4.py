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

# Debug flag for verbose output
DEBUG = os.environ.get("VITAL_DEBUG", "0") == "1"

# --- helper per normalitzar timestamps ---
def _ts_to_ms(ts_array):
    """Converteix timestamps (float segons) a int64 mil·lisegons per joins i CSV estables."""
    return (np.asarray(ts_array, dtype=np.float64) * 1000.0).round().astype(np.int64)


def process_segment(cfg, start_time, vf, arr_utils, all_data):
    interval_s = cfg['interval_secs']
    if os.environ.get("VITAL_TRACE") == "1":
        print(f"[TRACE] process_segment() iniciat amb start_time={start_time:.2f}s per {cfg.get('signal_track')}")

    # Millora: reutilitzar la senyal carregada per evitar recarregar-la múltiples vegades
    if all_data is None or all_data.shape[0] == 0:
        print(f"[ERROR] Debug: No es va passar la senyal per {cfg['signal_track']}")
        return None

    # Selecció per timestamps (més robust que aproximar amb sample_rate)
    ts_all = all_data[:, 0]
    # start_time i interval_s estan en segons absoluts; fem servir cerca binària
    start_idx = np.searchsorted(ts_all, start_time, side="left")
    end_idx = np.searchsorted(ts_all, start_time + interval_s, side="left")

    if end_idx - start_idx <= 0 or end_idx > len(ts_all):
        return None

    segment_data = all_data[start_idx:end_idx]
    if len(segment_data) == 0:
        return None

    timestamps = segment_data[:, 0]
    signal = segment_data[:, 1]

    signal = arr_utils.interp_undefined(signal)

    # --- Estimate original sampling rate from timestamps (robust) ---
    dt = np.diff(timestamps)
    if dt.size > 0:
        dt = dt[np.isfinite(dt) & (dt > 0)]
    orig_rate_est = float(np.median(1.0 / dt)) if dt.size > 0 else float(cfg.get('orig_rate', cfg['resample_rate']))

    target_rate = int(cfg['resample_rate'])
    target_len_cfg = int(cfg['signal_length'])
    expected_len = int(round(cfg['interval_secs'] * target_rate))
    if expected_len != target_len_cfg:
        print(f"[WARN] Config mismatch: interval_secs*resample_rate={expected_len} != signal_length={target_len_cfg}. Using signal_length as target length.")
        expected_len = target_len_cfg

    # --- Resample signal and timestamps to target rate ---
    signal = arr_utils.resample_hz(signal, orig_rate_est, target_rate)
    timestamps = arr_utils.resample_hz(timestamps, orig_rate_est, target_rate)

    # --- Enforce exact target length (pad/trim) ---
    cur_len = len(signal)
    if cur_len < expected_len:
        pad_n = expected_len - cur_len
        if cur_len > 0:
            # Edge padding requires no constant_values argument
            signal = np.pad(signal, (0, pad_n), mode='edge')
            timestamps = np.pad(timestamps, (0, pad_n), mode='edge')
        else:
            # Empty segment: fall back to constant padding
            pad_val_sig = 0.0
            pad_val_ts = timestamps[0] if len(timestamps) > 0 else start_time
            signal = np.pad(signal, (0, pad_n), mode='constant', constant_values=(pad_val_sig, pad_val_sig))
            timestamps = np.pad(timestamps, (0, pad_n), mode='constant', constant_values=(pad_val_ts, pad_val_ts))
    elif cur_len > expected_len:
        signal = signal[:expected_len]
        timestamps = timestamps[:expected_len]

    # Final safety check
    if len(signal) != expected_len:
        print(f"[ERROR] Debug: Longitud inesperada després del resampling: {len(signal)} != {expected_len}")
        return None

    inp = signal.reshape(1, 1, -1)
    try:
        pred_result = cfg['model'].predict(inp)
        pred_val = float(pred_result.squeeze())
        # Guardem el temps com a int64 (ms) per evitar problemes de join per flotants
        ts_ms = _ts_to_ms([timestamps[cfg['signal_length'] - 1]])[0]
        return {
            'Tiempo': ts_ms,
            cfg['output_var']: pred_val,
            '_valid': np.isfinite(pred_val),
            '_err': None,
            '_track': cfg.get('signal_track'),
        }
    except Exception as e:
        if os.environ.get("VITAL_TRACE") == "1" or DEBUG:
            print(f"[ERROR] Error en process_segment ({cfg.get('signal_track')}): {e}")
        # tornar un registre marcat com a error perquè puguem comptar-lo
        last_ts = timestamps[-1] if len(timestamps) > 0 else start_time
        ts_ms = _ts_to_ms([last_ts])[0]
        return {
            'Tiempo': ts_ms,
            cfg['output_var']: None,
            '_valid': False,
            '_err': str(e),
            '_track': cfg.get('signal_track'),
        }


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
        # MODIFICAT
        self.state_file = os.path.join(results_dir, "state_wave.json")
        if os.path.exists(self.state_file):
            import json
            try:
                with open(self.state_file, "r") as f:
                    self.last_processing_time = json.load(f)
                    if not isinstance(self.last_processing_time, dict):
                        raise ValueError("state_wave.json no és un dict")
            except Exception as e:
                print(f"[WARN] Estat invàlid a state_wave.json. Es reinicia: {e}")
                self.last_processing_time = {}

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
        # MODIFICAT
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
        chunk_duration = 180  # Duración en segundos para cada chunk (ampliado para procesar más por pasada)

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

            # === Durada i domini temporal reals (basat en timestamps del fitxer) ===
            ts_all = all_data_full[:, 0]
            if ts_all.size == 0:
                print(f"[WARN] Sense timestamps per {signal_track}")
                continue
            min_ts = float(ts_all[0])
            max_ts = float(ts_all[-1])

            interval_s = cfg['interval_secs']
            overlap_s = cfg['overlap_secs']
            step_s = max(1e-3, float(interval_s - overlap_s))

            # Generar start_times en el domini real (no assumim que comenci a 0)
            if max_ts - min_ts > interval_s:
                all_start_times = np.arange(min_ts, max_ts - interval_s, step_s, dtype=float)
            else:
                all_start_times = np.array([], dtype=float)
            if len(all_start_times) == 0:
                continue

            # Sincronització: fem servir timestamps reals per recordar el darrer processat
            file_key = os.path.basename(vital_path)
            last_processed = self.last_processing_time.get(file_key, min_ts - 1)

            batch_size = 100  # Procesar más segmentos por lote (si recursos lo permiten)

            if last_processed > min_ts:
                remaining_start_times = all_start_times[all_start_times > last_processed]
                if len(remaining_start_times) > 0:
                    start_times = remaining_start_times
                    print(f"[INFO] Continuando desde {last_processed:.2f}s reales, {len(start_times)} segmentos pendientes")
                else:
                    # No hay segmentos nuevos, procesar últimos batch_size
                    start_times = all_start_times[-batch_size:] if len(all_start_times) >= batch_size else all_start_times
                    print(f"[INFO] No hay segmentos nuevos, procesando últimos {len(start_times)} segmentos")
            else:
                # Primera ejecución, procesar todo
                start_times = all_start_times
                print(f"[INFO] Primera ejecución: {len(start_times)} segmentos disponibles en [{min_ts:.2f}, {max_ts:.2f}]s")

            if len(start_times) == 0:
                continue

            # -----------------------------------------------------------------------
            # PROCESSAMENT EN CHUNKS en domini real
            # -----------------------------------------------------------------------
            chunk_start_times = np.arange(min_ts, max_ts, chunk_duration, dtype=float)
            for chunk_start in chunk_start_times:
                chunk_end = chunk_start + chunk_duration + interval_s  # cobrir segments que travessen el límit
                # Filtrar start_times que caen dentro del chunk (domini real)
                chunk_segment_starts = start_times[(start_times >= chunk_start) & (start_times < chunk_end)]
                if len(chunk_segment_starts) == 0:
                    continue

                # Obtenir el subarray del rang temporal real
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

                # --- DEBUG: mètriques del chunk ---
                total = len(segment_results)
                errs = sum(1 for r in segment_results if r is not None and r.get('_err'))
                valids = [r for r in segment_results if r is not None and r.get('_valid')]
                n_valids = len(valids)
                n_nans = sum(1 for r in segment_results if r is not None and (r.get('_valid') is False) and (r.get('_err') is None))

                if DEBUG or os.environ.get("VITAL_TRACE") == "1":
                    print(f"[DEBUG] {cfg['signal_track']} chunk: segs={total}, valids={n_valids}, nans={n_nans}, errors={errs}")

                # Guardar-ho tot (també NaN i errors) per poder inspeccionar després
                clean = []
                for r in segment_results:
                    if r is None:
                        continue
                    rec = {k: v for k, v in r.items() if not k.startswith('_')}
                    clean.append(rec)
                records.extend(clean)
                gc.collect()

                # Actualizar último tiempo procesado para evitar reprocesamientos
                if len(chunk_segment_starts) > 0:
                    self.last_processing_time[file_key] = max(chunk_segment_starts)

                current_elapsed = time.time() - processing_start_time
                # Ampliem límit per permetre processar .vital grans d'una sola tirada
                if current_elapsed > 600.0:
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
        # --- Simplificació: exportem només les files amb predicció (sense join a la timeline completa) ---
        # Evitem buits per desajust de timestamps. Ens quedem amb df_preds, assegurant tipus/ordre.
        df = df_preds.with_columns(
        pd.col("Tiempo").cast(pd.Int64)
        ).sort("Tiempo").unique(subset=["Tiempo"], keep="last")
        # Assegurem que totes les columnes de sortida esperades existeixen encara que algun model no hagi retornat files
        wave_output_vars = [cfg['output_var'] for cfg in self.model_configs if cfg.get('input_type') == 'wave']
        if wave_output_vars:
            for var in wave_output_vars:
                if var not in df.columns:
                    # crea la columna com a nulls (Float64) per evitar ColumnNotFoundError en el filtre
                    df = df.with_columns(pd.lit(None).cast(pd.Float64).alias(var))

        # --- Debug: comptar files amb almenys una predicció no nul·la ---
        try:
            _pred_rows = 0
            if wave_output_vars:
                _pred_rows = df.select(
                    pd.any_horizontal([pd.col(var).is_not_null() for var in wave_output_vars]).alias("has_pred")
                ).filter(pd.col("has_pred")).height
            print(f"[DEBUG] Files amb alguna predicció no nul·la: {_pred_rows} / {df.height}")
        except Exception as _e:
            print(f"[WARN] No s'ha pogut calcular el recompte de files amb predicció: {_e}")

        # Guardem el dataframe complet per la UI
        self.latest_df = df.clone()

        csv_path = xlsx_path.replace(".xlsx", ".csv")

        if DEBUG:
            # En mode DEBUG no filtrem res: volem veure totes les files generades
            df_export = df.sort("Tiempo")
            # Escriu també una còpia crua per inspecció
            raw_path = xlsx_path.replace(".xlsx", "_raw.csv")
            self._save_csv(df_export, raw_path, first)
        else:
            # NOMÉS EXPORTAR FILES AMB ALGUNA PREDICCIÓ no nul·la
            wave_output_vars = [cfg['output_var'] for cfg in self.model_configs if cfg.get('input_type') == 'wave']
            df_export = df
            if wave_output_vars:
                df_export = df.filter(
                    pd.any_horizontal([pd.col(var).is_not_null() for var in wave_output_vars])
                )

        # ESCRIPTURA INCREMENTAL DEL CSV
        self._save_csv(df_export, csv_path, first)
        # Guardar estat de processament per wave
        import json
        with open(self.state_file, "w") as f:
            json.dump(self.last_processing_time, f)
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
            df = df.with_columns(pd.Series(name=cfg['output_var'], values=preds).cast(pd.Float64))
        return df
    # MODIFICAT
    def _save_csv(self, df: pd.DataFrame, path: str, first: bool):
        """
        Escritura incremental en CSV:
        - Si és la primera vegada, crea arxiu amb capçaleres.
        - Si no, afegeix només files noves (ordenades per "Tiempo") sense capçaleres.
        """
        if not isinstance(df, PLDataFrame):
            raise TypeError("Expected a Polars DataFrame")

        # Ordena i elimina duplicats per seguretat
        if "Tiempo" in df.columns:
            df = df.sort("Tiempo").unique(subset=["Tiempo"], keep="last")

        if first or not os.path.exists(path):
            df.write_csv(path)
            return

        # Mode append: intentem evitar duplicats llegint l'últim valor de Tiempo del fitxer existent
        try:
            # Llegim només l'última línia del CSV existent de forma eficient
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                pos = f.tell()
                # Retrocedeix fins trobar salt de línia
                while pos > 0:
                    pos -= 1
                    f.seek(pos, os.SEEK_SET)
                    if f.read(1) == b"\n":
                        break
                last_line = f.readline().decode(errors="ignore").strip()
            if last_line:
                # S'assumeix que la primera columna és Tiempo
                last_ts = last_line.split(",", 1)[0]
                try:
                    last_ts = int(last_ts)
                    df = df.filter(pd.col("Tiempo") > last_ts)
                except ValueError:
                    pass
        except Exception as e:
            print(f"[WARN] No s'ha pogut llegir l'última línia del CSV existent: {e}")

        if df.height == 0:
            return

        with open(path, "a") as f:
            df.write_csv(f, include_header=False)