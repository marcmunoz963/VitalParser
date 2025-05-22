import os
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from parser.vital_utils import is_nan, find_latest_vital
from vitaldb import VitalFile
import parser.arr as arr_utils

class VitalProcessor:
    def __init__(self, model_configs, results_dir, window_rows=20, poll_interval=1):
        """
        Añade wave_last para controlar tiempos de predicción por modelo wave.
        """
        self.model_configs = model_configs
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.window_rows = window_rows
        self.latest_df = None
        # Estado para limitar predicciones wave según interval_secs
        # clave: output_var, valor: timestamp de última predicción
        self.wave_last = {cfg.get('output_var'): None for cfg in model_configs if cfg.get('input_type')=='wave'}
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
        self.latest_df = None

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
        self._run_predictions(df)
        self.latest_df = df.copy()
        self._save_excel(df, xlsx_path, first)
        return df

    def _process_wave(self, recordings_dir):
        """
        Procesa pipelines de onda: carga latest.vital, aplica segmentación y modelo wave.
        Además, incluye en la salida los valores de todas las pistas en los mismos timestamps,
        y posiciona la columna de predicción al final.
        """
        vital_path = find_latest_vital(recordings_dir)
        if not vital_path:
            return None
        base = os.path.splitext(os.path.basename(vital_path))[0]
        xlsx_path = os.path.join(self.results_dir, f"{base}_wave.xlsx")
        first = not os.path.exists(xlsx_path)

        vf = VitalFile(vital_path)
        # Obtener todas las pistas para agregar sus valores
        all_tracks = vf.get_track_names()
        raw_all = vf.to_numpy(all_tracks, interval=0, return_timestamp=True)
        df_all = pd.DataFrame(raw_all, columns=['Tiempo'] + all_tracks)

        records = []
        for cfg in self.model_configs:
            if cfg.get('input_type') != 'wave':
                continue
            track       = cfg.get('signal_track')
            length      = cfg.get('signal_length', 2000)
            resample_hz = cfg.get('resample_rate', 100)
            interval_s  = cfg.get('interval_secs', length // resample_hz)
            overlap_s   = cfg.get('overlap_secs', interval_s // 2)
            out_var     = cfg.get('output_var')
            if not track:
                continue

            data = vf.to_numpy([track], interval=0, return_timestamp=True)
            if data is None or data.shape[0] == 0:
                continue
            timestamps = data[:, 0]
            signal     = data[:, 1]

            # Preprocesado
            signal = arr_utils.interp_undefined(signal)
            signal = arr_utils.resample_hz(signal, cfg.get('orig_rate', resample_hz), resample_hz)

            # Ventana final y predicción
            snippet = signal[-length:]
            if snippet.size < length:
                snippet = np.pad(snippet, (length - snippet.size, 0))
            inp = snippet.reshape(1, 1, -1)
            try:
                pred = float(cfg['model'].predict(inp).squeeze())
            except Exception as e:
                print(f"Prediction error (wave): {e}")
                continue

            time = timestamps[-1]
            records.append({'Tiempo': time, out_var: pred})

        if not records:
            return None

        # DataFrame de predicciones
        df_preds = pd.DataFrame(records)
        # Unir con df_all para incluir todas las variables
        df = pd.merge(df_all, df_preds, on='Tiempo', how='left')
        # Reordenar para mover la predicción al final
        cols = [c for c in df.columns if c not in df_preds.columns or c == 'Tiempo']
        # cols now has 'Tiempo' + all_tracks
        # append prediction column
        pred_col = [c for c in df_preds.columns if c != 'Tiempo'][0]
        cols = cols + [pred_col]
        df = df[cols]

        self.latest_df = df.copy()
        self._save_excel(df, xlsx_path, first)
        return df

    def _run_predictions(self, df: pd.DataFrame):
        for cfg in self.model_configs:
            if cfg.get('input_type') != 'tabular':
                continue
            ws    = cfg.get('window_size', 1)
            vars_ = cfg.get('input_vars', [])
            cols  = [c for c in vars_ if c in df]
            if not cols:
                continue
            arr = df[cols].fillna(0).values
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

            df[cfg['output_var']] = preds

    def _save_excel(self, df: pd.DataFrame, path: str, first: bool):
        if first:
            df.to_excel(path, index=False)
        else:
            wb = load_workbook(path)
            ws = wb.active
            for row in df.itertuples(index=False):
                ws.append(row)
            wb.save(path)
