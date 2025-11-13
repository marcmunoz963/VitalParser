# test_vital_to_zarr.py
# Llegeix un .vital, selecciona tracks, i escriu en un arxiu .zarr
# Estructura:
#   signals/<vendor>/<track>/{time_ms:int64, value:float32}
# Funciona en mode append i evita crear grups/datasets si no hi ha dades útils.

import os
import time
import numpy as np
import zarr
from numcodecs import Blosc
from vitaldb import VitalFile

# --- Compressor recomanat (equilibri velocitat/compressió) ---
_ZSTD = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)


def _safe_group(root: zarr.hierarchy.Group, path: str) -> zarr.hierarchy.Group:
    """Crea (si cal) i retorna el subgrup dins root (p.ex. 'signals/Intellivue/PLETH')."""
    parts = [p for p in path.split("/") if p]
    g = root
    for p in parts:
        g = g.require_group(p)
    return g


def _get_group_if_exists(root: zarr.hierarchy.Group, path: str):
    """Retorna el grup si existeix, o None si no existeix (no crea res)."""
    parts = [p for p in path.split("/") if p]
    g = root
    for p in parts:
        if p in g:
            obj = g[p]
            if isinstance(obj, zarr.hierarchy.Group):
                g = obj
            else:
                return None
        else:
            return None
    return g


def _append_1d(ds: zarr.core.Array, values: np.ndarray) -> None:
    """Append eficient a un dataset 1D."""
    if values.size == 0:
        return
    n0 = ds.shape[0]
    ds.resize(n0 + values.size)
    ds[n0:] = values


def vital_to_zarr(
    vital_path: str,
    zarr_path: str,
    tracks: list[str],
    window_secs: float | None = None,
    chunk_len: int = 30000,
) -> None:
    """
    vital_path: ruta a l'arxiu .vital
    zarr_path:  ruta a l'arxiu .zarr d'output (un sol contenidor)
    tracks:     llista de tracks a escriure (p.ex. ["Intellivue/PLETH", "Intellivue/ABP"])
    window_secs: si s'indica, guarda només els últims X segons de cada track
    chunk_len:  mida de chunk per als datasets (mostres per chunk)
    """
    if not os.path.exists(vital_path):
        raise FileNotFoundError(f"No s'ha trobat el .vital: {vital_path}")

    # Obrim el VitalFile només amb els tracks demanats (estalvia memòria/temps)
    vf = VitalFile(vital_path, tracks)

    # Obrim o creem el contenidor Zarr (mode "a" per permetre append)
    os.makedirs(os.path.dirname(zarr_path) or ".", exist_ok=True)
    root = zarr.open_group(zarr_path, mode="a")

    # Metadades útils a nivell root
    root.attrs.setdefault("schema", "v1")
    root.attrs.setdefault("created_by", "test_vital_to_zarr.py")
    root.attrs.setdefault("time_origin", "epoch1700_ms")  # Vital: segons des de 1700-01-01; aquí guardem ms
    root.attrs["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

    signals_root = _safe_group(root, "signals")

    written_any = False
    total_added_samples = 0
    written_tracks = 0
    skipped_no_new = 0
    skipped_all_nan = 0
    skipped_empty = 0

    for track in tracks:
        # 1) Llegim del .vital
        try:
            data = vf.to_numpy([track], interval=0, return_timestamp=True)
        except Exception as e:
            print(f"[WARN] No s'ha pogut llegir el track '{track}': {e}")
            skipped_empty += 1
            continue

        if data is None or data.size == 0:
            print(f"[INFO] Track buit o inexistent al fitxer: {track}")
            skipped_empty += 1
            continue

        ts = data[:, 0].astype(np.float64, copy=False)
        vals = data[:, 1].astype(np.float64, copy=False)

        # 2) Finestra temporal opcional
        if window_secs is not None:
            t1 = ts[-1]
            t0 = t1 - float(window_secs)
            m = (ts >= t0) & (ts <= t1)
            ts = ts[m]
            vals = vals[m]

        if ts.size == 0:
            print(f"[INFO] Sense dades a la finestra per {track}")
            skipped_empty += 1
            continue

        ts_ms = np.rint(ts * 1000.0).astype(np.int64)
        vals_f32 = vals.astype(np.float32)

        # 3) Abans de crear res, mirem si el grup ja existeix per deduplicar
        track_group_path = f"signals/{track}"
        existing_grp = _get_group_if_exists(root, track_group_path)

        if existing_grp is not None and "time_ms" in existing_grp:
            try:
                last_ts = int(existing_grp["time_ms"][-1])
                mask_new = ts_ms > last_ts
                ts_ms = ts_ms[mask_new]
                vals_f32 = vals_f32[mask_new]
            except Exception as e:
                print(f"[WARN] No s'ha pogut llegir l'últim time_ms de '{track}': {e}")

        # 4) Si després del filtre no queda res, SALTEM i NO creem res
        if ts_ms.size == 0:
            print(f"[INFO] Res nou per {track}")
            skipped_no_new += 1
            continue

        # 5) Si tot el que queda és NaN, SALTEM i NO creem res
        if np.all(np.isnan(vals_f32)):
            print(f"[SKIP] {track}: totes les mostres noves són NaN")
            skipped_all_nan += 1
            continue

        # 6) Ara sí: creem el grup/datasets i fem append
        grp = _safe_group(signals_root, track)

        ds_time = grp.require_dataset(
            "time_ms",
            shape=(0,),
            chunks=(chunk_len,),
            dtype="int64",
            compressor=_ZSTD,
        )
        ds_val = grp.require_dataset(
            "value",
            shape=(0,),
            chunks=(chunk_len,),
            dtype="float32",
            compressor=_ZSTD,
        )

        _append_1d(ds_time, ts_ms)
        _append_1d(ds_val, vals_f32)

        grp.attrs["track"] = track
        grp.attrs.setdefault("units", "")
        grp.attrs.setdefault("notes", "")

        print(f"[OK] {track}: +{ts_ms.size} mostres (total={ds_time.shape[0]})")
        total_added_samples += int(ts_ms.size)
        written_tracks += 1
        written_any = True

    # 7) Resum
    if written_any:
        print(f"✅ Escrita/actualitzada la col·lecció a: {zarr_path}")
        print(f"   Resum: tracks escrites/actualitzades = {written_tracks}, mostres afegides = {total_added_samples}")
        print(f"   Saltats sense novetats = {skipped_no_new}, saltats tot NaN = {skipped_all_nan}, buits/inexistents = {skipped_empty}")
    else:
        print("⚠️  No s'ha escrit cap mostra nova (tracks inexistents, sense novetats o tot NaN).")


if __name__ == "__main__":
   # --- Ruta directa al fitxer .vital ---
    from config import VITAL_PATH, ZARR_PATH

    vital_path = VITAL_PATH
    zarr_path = ZARR_PATH  


    tracks = [
        # --- Senyals fisiològiques bàsiques ---
        "Intellivue/PLETH",
        "Intellivue/ABP",
        "Intellivue/ABP_SYS",
        "Intellivue/ABP_DIA",
        "Intellivue/ABP_MEAN",
        "Intellivue/ECG_I",
        "Intellivue/ECG_II",
        "Intellivue/ECG_III",
        "Intellivue/RESP",
        "Intellivue/RR",
        "Intellivue/HR",
        "Intellivue/SPO2",
        "Intellivue/PLETH_SAT_O2",

        # --- Derivacions ECG addicionals ---
        "Intellivue/ST_I",
        "Intellivue/ST_II",
        "Intellivue/ST_III",
        "Intellivue/ST_AVR",
        "Intellivue/ST_AVL",
        "Intellivue/ST_AVF",
        "Intellivue/ST_V",
        "Intellivue/ST_MCL",

        # --- Variables cerebrals ---
        "Intellivue/EEG_BIS",
        "Intellivue/EEG_RATIO_SUPPRN",
        "Intellivue/EEG_BIS_SQI",
        "Intellivue/EEG_BIS_ASYM",
        "Intellivue/EMG_ELEC_POTL_MUSCL",

        # --- Pressions intracranials / perfusió ---
        "Intellivue/ICP",
        "Intellivue/ICP_MEAN",
        "Intellivue/PRESS_CEREB_PERF",

        # --- CO₂ i gasometria ---
        "Intellivue/CO2",
        "Intellivue/CO2_MEAN",
        "Intellivue/CO2_ETCO2",
        "Intellivue/O2_FLOW",

        # --- Cardiovasculars avançats ---
        "Intellivue/CVP",
        "Intellivue/PPV",
        "Intellivue/CO",
        "Intellivue/SVV",
        "Intellivue/ART_HR",
        "Intellivue/ART_MAP",
        "Intellivue/ART_dPdtMax",

        # --- Nivells i alarms ---
        "Intellivue/ALARM",
        "Intellivue/ECG_VPC_CNT",
    ]

    print("[DEBUG] vital_to_zarr està definit:", callable(globals().get("vital_to_zarr")))
    vital_to_zarr(vital_path, zarr_path, tracks, window_secs=None)
