import os
import argparse
import numpy as np
import zarr
from numcodecs import Blosc
from datetime import datetime, timezone

# ------------------------------------------------------------
# Config CLI: pacient i data (YYYYMMDD). Si no es passen, per defecte.
# ------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--patient_id", default="BOX01", help="Identificador del box/pacient (p.ex. BOX12)")
parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"),
                    help="Data del registre en format YYYYMMDD (per rotació diària)")
parser.add_argument("--results_dir", default="results", help="Carpeta base on desar els .zarr")
args = parser.parse_args()

patient_id = args.patient_id
date_str = args.date
results_dir = args.results_dir

os.makedirs(results_dir, exist_ok=True)

# Fitxer Zarr ÚNIC per pacient i dia (rotació diària)
store_path = os.path.join(results_dir, f"{patient_id}_{date_str}.zarr")
root = zarr.open(store_path, mode="a")  # append/crea si no existeix

# Metadades bàsiques (escrivim-les un cop; s'actualitzen si cal)
root.attrs["patient_id"] = patient_id
root.attrs["date"] = date_str
root.attrs["tz"] = "UTC"
root.attrs["schema_version"] = "1.0"

# Grups lògics
signals_grp = root.require_group("signals")   # senyals originals (monitorització)
pred_grp    = root.require_group("pred")      # sortides d’algoritmes (prediccions, índexs, etc.)

# Compressor i chunking (Zarr v2)
compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
# Chunk de ~60k mostres (~5 min a 200 Hz). Ajusta-ho si cal.
chunks = (60_000,)

def get_or_create_1d(group, name, dtype="f4", fill=np.nan):
    """
    Obté (o crea si no existeix) un array 1D redimensionable amb Zarr v2.
    No fa cap 'shape check' agressiu, així evitarem errors de "shape do not match".
    """
    if name in group:
        return group[name]
    return group.create_dataset(
        name,
        shape=(0,),
        dtype=dtype,
        chunks=chunks,
        compressor=compressor,
        fill_value=fill,
        overwrite=False,
        maxshape=(None,),
    )

def append_1d(arr, data):
    """Afegim dades al final (append) amb un resize + slicing."""
    n_old = arr.shape[0]
    n_new = n_old + int(data.shape[0])
    arr.resize(n_new)
    arr[n_old:n_new] = data

def get_or_create_signal_pair(parent_group, signal_path, dtype="f4"):
    """
    Crea o obté el parell (time_ms, value) per una variable concreta.
    Ex.: signal_path="Intellivue/PLETH" -> crea subgrup "Intellivue", arrays "PLETH_time_ms" i "PLETH".
    """
    # Descomponem la ruta en grup i nom de senyal
    parts = signal_path.split("/")
    *grp_parts, var_name = parts
    g = parent_group
    for p in grp_parts:
        g = g.require_group(p)

    time_arr = get_or_create_1d(g, f"{var_name}_time_ms", dtype="i8", fill=-1)
    data_arr = get_or_create_1d(g, f"{var_name}", dtype=dtype, fill=np.nan)
    return time_arr, data_arr

# -------------------------------------------------------------------
# EXEMPLE: simulació d’escriptura (com si llegissis del .vital)
# En el teu pipeline real, substitueix aquesta part per valors reals
# -------------------------------------------------------------------
# Simulem 10.000 mostres de PLETH a 100 Hz i ART a 125 Hz
fs_pleth = 100.0
fs_art   = 125.0
n_pleth  = 10_000
n_art    = 12_500

# Temps d'inici "ara" (UTC) en mil·lisegons
t0_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

pleth_vals = (np.random.rand(n_pleth).astype("f4") * 1.0)
pleth_ts   = (t0_ms + (np.arange(n_pleth) * (1000.0 / fs_pleth))).astype("i8")

art_vals   = (80 + 40*np.sin(np.linspace(0, 10*np.pi, n_art))).astype("f4")
art_ts     = (t0_ms + (np.arange(n_art) * (1000.0 / fs_art))).astype("i8")

# Obtenim/creem els datasets on escriure
pleth_time_arr, pleth_arr = get_or_create_signal_pair(signals_grp, "Intellivue/PLETH", dtype="f4")
art_time_arr,   art_arr   = get_or_create_signal_pair(signals_grp, "Intellivue/ART",   dtype="f4")

# Append (afegeix al final del dia)
append_1d(pleth_time_arr, pleth_ts)
append_1d(pleth_arr,      pleth_vals)

append_1d(art_time_arr, art_ts)
append_1d(art_arr,      art_vals)

print(f"[OK] Escrit a {store_path}")
print("PLETH ->", pleth_arr.shape, "mostres  |  ART ->", art_arr.shape)

# -------------------------------------------------------------------
# OPCIONAL: lectura d’un interval temporal (en ms) per comprovar
# -------------------------------------------------------------------
def read_interval(group, signal_path, t_start_ms, t_end_ms):
    """Retorna (ts, vals) de signal_path dins [t_start_ms, t_end_ms]."""
    t_arr, v_arr = get_or_create_signal_pair(group, signal_path)  # només els obre si existeixen
    # Obtenim tot i filtrem (per simplicitat; amb dask/xarray ho pots fer lazy)
    ts = np.asarray(t_arr[:])
    vs = np.asarray(v_arr[:])
    if ts.size == 0:
        return np.array([], dtype="i8"), np.array([], dtype="f4")
    mask = (ts >= t_start_ms) & (ts <= t_end_ms)
    return ts[mask], vs[mask]

# Ex.: llegim els pròxims 3 segons de PLETH
t_start = t0_ms
t_end   = t0_ms + 3000
ts_seg, vals_seg = read_interval(signals_grp, "Intellivue/PLETH", t_start, t_end)
print(f"Segment PLETH {len(vals_seg)} mostres dins {t_start}–{t_end} ms")

import zarr, os

store_path = os.path.join("results", f"{patient_id}_{date_str}.zarr")
root = zarr.open(store_path, mode="r")

print("[ARBRE ZARR]")
def walk(g, prefix=""):
    for k, v in g.items():
        if isinstance(v, zarr.hierarchy.Group):
            print(prefix + f"[G] {k}/")
            walk(v, prefix + "  ")
        else:
            print(prefix + f"[A] {k}  shape={v.shape}  dtype={v.dtype}")

walk(root)