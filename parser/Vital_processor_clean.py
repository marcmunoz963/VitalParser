# -*- coding: utf-8 -*-
"""
VitalProcessorClean
-------------------
Objectiu: pipeline simple i robust per a models de WAVE que escriuen a CSV en mode append.
- Carrega només les tracks necessàries.
- Segmenta per timestamps reals (no per índexs).
- Resampleja de forma robusta (estimant la freq. d’origen amb els timestamps).
- Evita reprocessar segments (state per fitxer).
- Escriu incrementalment al CSV (sense reescriure-ho tot).
- Salta models sense carregar (cfg['model'] is None) sense trencar el flux.

Dependències: numpy, polars, vitaldb, parser.arr, parser.vital_utils
"""

import os
import json
import gc
import time
import numpy as np
import polars as pl
from polars import DataFrame as PLDataFrame
from concurrent.futures import ThreadPoolExecutor
from vitaldb import VitalFile
import parser.arr as arr_utils

from parser.vital_utils import find_latest_vital

DEBUG = os.environ.get("VITAL_DEBUG", "0") == "1"
TRACE = os.environ.get("VITAL_TRACE", "0") == "1"

# --- Tracks de tendència candidates (noms típics d’Intellivue) ---
TREND_CANDIDATES = [
    # Hemodinàmiques (IBP)
    "Intellivue/ABP_SYS",
    "Intellivue/ABP_DIA",
    "Intellivue/ABP_MEAN",
    "Intellivue/ABP_HR",

    # ECG
    "Intellivue/ECG_HR",
    "Intellivue/ECG_VPC_CNT",  # comptes de VPC
    "Intellivue/ST_I",
    "Intellivue/ST_II",
    "Intellivue/ST_III",
    "Intellivue/ST_AVR",
    "Intellivue/ST_AVL",
    "Intellivue/ST_AVF",
    "Intellivue/ST_V",
    "Intellivue/ST_MCL",

    # Respiratori
    "Intellivue/RR",

    # Oximetria/pleth
    "Intellivue/PLETH_SAT_O2",
    "Intellivue/PLETH_HR",
    "Intellivue/PLETH_PERF_REL",

    # EEG/BIS
    "Intellivue/EEG_BIS",
    "Intellivue/EEG_BIS_SQI",
    "Intellivue/EEG_BIS_ASYM",

    # Neurocritico
    "Intellivue/ICP_MEAN",
    "Intellivue/PRESS_CEREB_PERF",  # CPP
]


# --------- Helpers ---------
def _ts_to_ms(ts_array):
    """Converteix timestamps (segons float) a int64 (ms) per estabilitat en CSV."""
    return (np.asarray(ts_array, dtype=np.float64) * 1000.0).round().astype(np.int64)


def _estimate_orig_rate(ts_arr):
    """Estima la freqüència d’origen a partir dels timestamps (en segons)."""
    if ts_arr is None or len(ts_arr) < 2:
        return None
    dt = np.diff(ts_arr)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return None
    return float(np.median(1.0 / dt))


def _save_csv_incremental(df: PLDataFrame, path: str):
    """Escriu en mode append. Si no existeix, crea el CSV amb capçaleres."""
    if df is None or df.height == 0:
        return
    if not isinstance(df, PLDataFrame):
        raise TypeError("Expected a Polars DataFrame")

    # Ordenar per Tiempo i deduplicar
    if "Tiempo" in df.columns:
        df = df.sort("Tiempo").unique(subset=["Tiempo"], keep="last")
    # Garanteix que "Tiempo" és la primera columna al CSV
    cols = df.columns
    if "Tiempo" in cols:
        other = [c for c in cols if c != "Tiempo"]
        df = df.select(["Tiempo"] + other)

    first = not os.path.exists(path)
    if first:
        df.write_csv(path)
        return

    # Llegeix l’últim Tiempo del CSV existent per evitar duplicats
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell()
            while pos > 0:
                pos -= 1
                f.seek(pos, os.SEEK_SET)
                if f.read(1) == b"\n":
                    break
            last_line = f.readline().decode(errors="ignore").strip()
        if last_line:
            last_ts_str = last_line.split(",", 1)[0]
            last_ts = int(last_ts_str)
            df = df.filter(pl.col("Tiempo") > last_ts)
    except Exception as e:
        if DEBUG:
            print(f"[WARN] No s'ha pogut llegir l'última línia de {path}: {e}")

    if df.height == 0:
        return

    with open(path, "a") as f:
        df.write_csv(f, include_header=False)


# Helper to print not-null summary for debugging
def _nn_summary(df: pl.DataFrame, tag: str):
    if df is None:
        print(f"[DEBUG] {tag}: None")
        return
    try:
        nn = {c: int(df.select(pl.col(c).is_not_null().sum()).item()) for c in df.columns}
        print(f"[DEBUG] {tag} not-nulls:", {k: v for k, v in nn.items() if k != "Tiempo"})
    except Exception as e:
        print(f"[DEBUG] {tag} summary error: {e}")



# --- Tendències natives: recull automàtic de tracks rellevants i resampleig a 1 Hz ---
def _collect_raw_trends(vf: VitalFile, available_tracks: set, resample_hz: float = 1.0) -> pl.DataFrame | None:
    """
    Recull automàticament tendències disponibles entre TREND_CANDIDATES,
    les resampleja (si cal) a ~1 Hz i retorna un DataFrame amb Tiempo (ms) + columnes.
    """
    # Selecció dels candidats que realment existeixen al fitxer
    wanted = [t for t in TREND_CANDIDATES if t in available_tracks]
    if not wanted:
        return None

    # Carrega totes les sèries en bloc (timestamps + valors per track)
    # vitaldb.to_numpy([...], interval=0, return_timestamp=True) retorna:
    #  N x (1 + len(tracks))  [ts, v1, v2, ...]
    try:
        arr = vf.to_numpy(wanted, interval=0, return_timestamp=True)
        if arr is None or arr.shape[0] == 0:
            return None
    except Exception:
        return None

    ts = arr[:, 0].astype(float)
    vals = arr[:, 1:]  # columnes en el mateix ordre que 'wanted'

    # Estimem freq original i, si cal, fem resample a ~1 Hz per compactar
    dt = np.diff(ts)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    orig_hz = float(np.median(1.0 / dt)) if dt.size else None

    if orig_hz is not None and orig_hz > (resample_hz + 1e-6):
        # Resample fem servir els timestamps com a senyal guia
        # Creem una malla temporal uniformement espaiada a 1/resample_hz
        t0, t1 = ts[0], ts[-1]
        if t1 <= t0:
            return None
        step = 1.0 / resample_hz
        ts_new = np.arange(t0, t1 + 1e-9, step, dtype=float)

        # Interpolem cada columna de vals a ts_new (interpolació lineal simple per buits petits)
        out_cols = []
        for i in range(vals.shape[1]):
            vi = vals[:, i]
            # Eliminem punts no finits per a la interpolació
            mask = np.isfinite(vi) & np.isfinite(ts)
            if mask.sum() < 2:
                # massa pocs punts bons → tot NaN
                out_cols.append(np.full_like(ts_new, np.nan, dtype=float))
                continue
            out_cols.append(np.interp(ts_new, ts[mask], vi[mask], left=np.nan, right=np.nan))

        ts_ms = _ts_to_ms(ts_new)
        data = {"Tiempo": ts_ms}
        for name, col in zip(wanted, out_cols):
            # usem noms curts per llegibilitat (última part després de '/')
            short = name.split("/")[-1]
            data[short] = col
        return pl.DataFrame(data)

    else:
        # Sense resample (ja és lent o irregular): passem tal qual
        ts_ms = _ts_to_ms(ts)
        data = {"Tiempo": ts_ms}
        for j, name in enumerate(wanted):
            short = name.split("/")[-1]
            data[short] = vals[:, j].astype(float)
        # Deduplica per Tiempo i conserva l’últim
        return pl.DataFrame(data).sort("Tiempo").unique(subset=["Tiempo"], keep="last")


# --------- Segment processing ---------
def _process_segment(cfg, start_time, all_data_chunk):
    """
    Processa un sol segment per un model wave.
    - cfg: dict del model (signal_track, resample_rate, signal_length, interval_secs, output_var, model)
    - start_time: inici (segons reals) del segment dins del fitxer
    - all_data_chunk: np.array N x 2 [timestamp(sec), value] d’un subrang que cobreix el segment
    """
    if cfg.get("model") is None:
        if TRACE:
            print(f"[SKIP] {cfg.get('name','(sense nom)')} sense model carregat")
        return None

    interval_s = float(cfg["interval_secs"])

    if all_data_chunk is None or all_data_chunk.shape[0] == 0:
        if DEBUG:
            print(f"[ERROR] Sense dades al chunk per {cfg['signal_track']}")
        return None

    ts_all = all_data_chunk[:, 0]
    # Troba rang del segment dins del chunk (per timestamps reals)
    start_idx = np.searchsorted(ts_all, start_time, side="left")
    end_idx = np.searchsorted(ts_all, start_time + interval_s, side="left")
    if end_idx - start_idx <= 0 or end_idx > len(ts_all):
        return None

    seg = all_data_chunk[start_idx:end_idx]
    if seg.size == 0:
        return None

    timestamps = seg[:, 0]
    signal = seg[:, 1]

    # neteja/interpolació
    signal = arr_utils.interp_undefined(signal)

    # resampleig robust
    orig_rate = _estimate_orig_rate(timestamps) or float(cfg.get("resample_rate"))
    target_rate = int(cfg["resample_rate"])
    target_len_cfg = int(cfg["signal_length"])
    expected_len = int(round(float(cfg["interval_secs"]) * target_rate))

    # Si hi ha discrepància, fem servir signal_length com a font de veritat
    if expected_len != target_len_cfg and DEBUG:
        print(f"[WARN] Mismatch: interval*rate={expected_len} != signal_length={target_len_cfg}. S'usarà signal_length.")
    expected_len = target_len_cfg

    signal_rs = arr_utils.resample_hz(signal, orig_rate, target_rate)
    # Construïm timestamps uniformes per a la finestra resamplejada
    if len(timestamps) > 0:
        ts_rs = np.linspace(float(timestamps[0]), float(timestamps[-1]), expected_len, dtype=float)
    else:
        ts_rs = np.linspace(start_time, start_time + float(cfg["interval_secs"]), expected_len, dtype=float)

    # Ajusta longitud exacta
    cur_len = len(signal_rs)
    if cur_len < expected_len:
        pad = expected_len - cur_len
        if cur_len > 0:
            signal_rs = np.pad(signal_rs, (0, pad), mode="edge")
            ts_rs = np.pad(ts_rs, (0, pad), mode="edge")
        else:
            # segment buit → padding constant
            pad_val_sig = 0.0
            pad_val_ts = timestamps[0] if len(timestamps) > 0 else start_time
            signal_rs = np.pad(signal_rs, (0, pad), mode="constant", constant_values=(pad_val_sig, pad_val_sig))
            ts_rs = np.pad(ts_rs, (0, pad), mode="constant", constant_values=(pad_val_ts, pad_val_ts))
    elif cur_len > expected_len:
        signal_rs = signal_rs[:expected_len]
        ts_rs = ts_rs[:expected_len]

    if len(signal_rs) != expected_len:
        if DEBUG:
            print(f"[ERROR] Longitud inesperada: {len(signal_rs)} != {expected_len}")
        return None

    try:
        x = signal_rs.reshape(1, 1, -1)
        y = cfg["model"].predict(x)
        pred_val = float(np.asarray(y).squeeze())
        ts_ms = _ts_to_ms([ts_rs[-1]])[0]
        return {"Tiempo": ts_ms, cfg["output_var"]: pred_val}
    except Exception as e:
        if DEBUG or TRACE:
            print(f"[ERROR] Predict {cfg.get('name','(sense nom)')}: {e}")
        ts_ms = _ts_to_ms([ts_rs[-1] if len(ts_rs) else start_time])[0]
        return {"Tiempo": ts_ms, cfg["output_var"]: None}


# --------- Processador net ---------
class VitalProcessor:
    """
    Versió neta i minimalista (WAVE → CSV). Sense Excel. Sense “tabular”.
    """
    def __init__(self, model_configs, results_dir, poll_interval=1):
        self.model_configs = model_configs
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.poll_interval = poll_interval
        self.latest_df = None
        # Estat per no repetir segments (clau: nom fitxer .vital → timestamp real últim processat)
        self.state_path = os.path.join(self.results_dir, "state_wave.json")
        self.state = self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r") as f:
                    st = json.load(f)
                return st if isinstance(st, dict) else {}
            except Exception:
                return {}
        return {}

    def _save_state(self):
        try:
            with open(self.state_path, "w") as f:
                json.dump(self.state, f)
        except Exception:
            pass

    def process_once(self, recordings_dir, mode="wave"):
        if mode != "wave":
            raise ValueError("Aquesta versió neta només implementa mode 'wave'.")
        return self._process_wave(recordings_dir)

    def _process_wave(self, recordings_dir):
        """
        Flux:
          1) Obté el .vital més recent.
          2) Determina quines tracks calen (i existeixen).
          3) Per cada model 'wave':
              - Construeix la graella de start_times (reals) amb solapament.
              - Processa en “chunks temporals” i en paral·lel.
          4) Escriu resultats (només prediccions) a CSV incremental.
        """
        vital_path = find_latest_vital(recordings_dir)
        if not vital_path:
            print("[ERROR] No s'ha trobat cap .vital")
            return None

        base = os.path.splitext(os.path.basename(vital_path))[0]
        csv_path = os.path.join(self.results_dir, f"{base}_wave.csv")
        processing_start = time.time()

        # 1) Calc tracks disponibles i quines calen
        vf_hdr = VitalFile(vital_path)  # header/només
        all_tracks = set(vf_hdr.get_track_names())

        wave_cfgs = [
            cfg for cfg in self.model_configs
            if cfg.get("input_type") == "wave"
        ]

        # Filtra també els que realment existeixen al fitxer:
        wave_cfgs = [
            cfg for cfg in wave_cfgs
            if cfg.get("signal_track") in all_tracks
        ]

        if len(wave_cfgs) == 0:
            print("[WARN] No hi ha models 'wave' aplicables a les tracks d'aquest .vital.")
            return None

        # 2) Carrega cache de dades per track (una sola vegada)
        #    Guardem N x 2 (timestamp(sec), value)
        data_cache = {}
        vf = VitalFile(vital_path, list({cfg["signal_track"] for cfg in wave_cfgs}))
        for cfg in wave_cfgs:
            tr = cfg["signal_track"]
            try:
                if tr not in data_cache:
                    arr = vf.to_numpy([tr], interval=0, return_timestamp=True)
                    if arr is None or arr.shape[0] == 0:
                        if DEBUG: print(f"[WARN] Sense dades per {tr}")
                        continue
                    data_cache[tr] = arr
                    if DEBUG:
                        print(f"[DEBUG] {tr} carregat: {arr.shape[0]} mostres, "
                              f"rang {arr[0,0]:.2f}–{arr[-1,0]:.2f}s")
            except Exception as e:
                if DEBUG: print(f"[ERROR] Carregant {tr}: {e}")

        records = []
        file_key = os.path.basename(vital_path)
        # Permet reiniciar l'estat per a aquest fitxer via variable d'entorn
        if os.environ.get("VITAL_RESET_STATE") == "1":
            self.state.pop(file_key, None)
        last_processed = float(self.state.get(file_key, -1e9))
        if DEBUG:
            print(f"[DEBUG] state_path={self.state_path}  file_key={file_key}  last_processed={last_processed:.2f}")

        # 3) Per cada model, genera start_times reals i processa en chunks
        #    Chunk temporal gran (en segons reals) per limitar memòria i aprofitar paral·lelisme
        chunk_span = 180.0

        for cfg in wave_cfgs:
            tr = cfg["signal_track"]
            model = cfg.get("model")
            if model is None:
                if TRACE:
                    print(f"[SKIP] {cfg.get('name','(sense nom)')} sense model")
                continue
            if tr not in data_cache:
                continue

            all_data = data_cache[tr]
            ts = all_data[:, 0]
            if ts.size == 0:
                continue
            t0, t1 = float(ts[0]), float(ts[-1])

            interval_s = float(cfg["interval_secs"])
            overlap_s  = float(cfg["overlap_secs"])
            step_s = max(1e-3, interval_s - overlap_s)
            if DEBUG:
                print(f"[DEBUG] {tr}: t0={t0:.2f}, t1={t1:.2f}, dur={t1 - t0:.2f}s, interval={interval_s}s, overlap={overlap_s}s, step={step_s}s")

            if (t1 - t0) <= interval_s:
                continue

            all_starts = np.arange(t0, t1 - interval_s, step_s, dtype=float)
            if DEBUG:
                print(f"[DEBUG] {tr}: all_starts inicials = {all_starts.size}")

            # Evita reprocessar (si ja vam arribar més lluny)
            if last_processed > t0:
                all_starts = all_starts[all_starts > last_processed]
            if DEBUG:
                print(f"[DEBUG] {tr}: all_starts post-state = {all_starts.size} (last_processed={last_processed:.2f})")

            if all_starts.size == 0:
                # Fallback: processar els darrers N segments per assegurar sortida
                N = 100  # ajustable
                if (t1 - t0) > interval_s:
                    step_s = max(1e-3, interval_s - overlap_s)
                    span_needed = interval_s + step_s * max(1, N - 1)
                    start_min = max(t0, t1 - span_needed)
                    fallback_starts = np.arange(start_min, max(t0, t1 - interval_s), step_s, dtype=float)
                    if fallback_starts.size > 0:
                        if DEBUG:
                            print(f"[INFO] {tr}: no hi ha segments nous; processant fallback {fallback_starts.size} darrers segments")
                        all_starts = fallback_starts
                    else:
                        if DEBUG:
                            print(f"[INFO] Res nou per {tr} (ni tan sols fallback)")
                        continue
                else:
                    if DEBUG:
                        print(f"[INFO] Res nou per {tr} (durada <= interval)")
                    continue

            # Processa per finestres grans de temps real
            chunk_edges = np.arange(t0, t1 + chunk_span, chunk_span)
            for cs in chunk_edges:
                ce = cs + chunk_span + interval_s  # cobrir segments que creuen límit
                starts_chunk = all_starts[(all_starts >= cs) & (all_starts < ce)]
                if starts_chunk.size == 0:
                    continue

                # Subarray amb dades del rang [cs, ce)
                mask = (all_data[:, 0] >= cs) & (all_data[:, 0] < ce)
                data_chunk = all_data[mask]
                if data_chunk is None or data_chunk.shape[0] == 0:
                    continue

                # Paral·lel: 1/2 CPUs
                n_workers = max(1, (os.cpu_count() or 2) // 2)
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    futs = [ex.submit(_process_segment, cfg, st, data_chunk) for st in starts_chunk]
                    out = [f.result() for f in futs]

                # Neteja i recull
                for r in out:
                    if r is None:
                        continue
                    records.append(r)

                # Avança el marcador de progrés
                self.state[file_key] = float(starts_chunk.max())

                # Límits de seguretat per a la GUI: talla passades molt llargues
                if (time.time() - processing_start) > 12.0 and len(records) > 5000:
                    break

        # --- 4) DataFrames de sortida ---

        # 4.a) Tendències natives (sempre que existeixin). Són la base.
        df_raw = _collect_raw_trends(vf_hdr, all_tracks)
        if DEBUG:
            _nn_summary(df_raw, "RAW(trends)")

        # 4.b) Prediccions (no filtris encara; podrien ser None al principi)
        df_pred = pl.DataFrame(records) if records else None
        if df_pred is not None and "Tiempo" in df_pred.columns:
            df_pred = df_pred.with_columns(pl.col("Tiempo").cast(pl.Int64))
            # dedup per Tiempo
            df_pred = df_pred.sort("Tiempo").unique(subset=["Tiempo"], keep="last")
        if DEBUG and df_pred is not None:
            _nn_summary(df_pred, "PRED(raw)")

        # 4.c) Compon la taula final:
        #     - Si hi ha tendències, parteix d'elles i fes outer join amb prediccions.
        #     - Si no hi ha tendències, exporta igualment les prediccions (encara que siguin None).
        if df_raw is not None and df_pred is not None and df_pred.height > 0:
            df_out = df_raw.join(df_pred, on="Tiempo", how="outer")
        elif df_raw is not None:
            df_out = df_raw
        elif df_pred is not None:
            df_out = df_pred
        else:
            if DEBUG:
                print("[WARN] Sense dades de tendència ni prediccions per exportar.")
            return None

        # 4.d) Neteja de duplicats de clau i ordre de columnes
        if "Tiempo_right" in df_out.columns:
            if "Tiempo" in df_out.columns:
                df_out = df_out.drop("Tiempo_right")
            else:
                df_out = df_out.rename({"Tiempo_right": "Tiempo"})

        if "Tiempo" in df_out.columns:
            df_out = df_out.sort("Tiempo").unique(subset=["Tiempo"], keep="last")
            # Assegura 'Tiempo' com a primera columna
            other = [c for c in df_out.columns if c != "Tiempo"]
            df_out = df_out.select(["Tiempo"] + other)

        # 4.e) Evita files totalment buides (cap columna amb valor)
        cols_no_tiempo = [c for c in df_out.columns if c != "Tiempo"]
        if cols_no_tiempo:
            df_out = df_out.filter(pl.any_horizontal([pl.col(c).is_not_null() for c in cols_no_tiempo]))

        if DEBUG:
            _nn_summary(df_out, "OUT(final)")

        # 5) Escriu CSV incremental i guarda estat
        _save_csv_incremental(df_out, csv_path)
        self.latest_df = df_out.clone()
        self._save_state()
        return self.latest_df
