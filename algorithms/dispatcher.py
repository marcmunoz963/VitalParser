# dispatcher.py
from __future__ import annotations

import os
import sys
import zarr
import numpy as np

# Afegim l'arrel del projecte (VitalParser) al sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from typing import Dict, List, Any

from cardiac_output import CardiacOutput, REQUIRED_TRACKS as CO_REQUIRED_TRACKS
from parser.test_vital_to_zarr import vital_to_zarr



# 🔧 CONFIG: rutes per defecte RELATIVES a l'arrel del projecte
DEFAULT_VITAL_REL = "records/3/250127/n5j8vrrsb_250127_100027.vital"
DEFAULT_ZARR_REL = "results/test_alg.zarr"

# Les convertim a rutes absolutes basades en PROJECT_ROOT
DEFAULT_VITAL = os.path.join(PROJECT_ROOT, DEFAULT_VITAL_REL)
DEFAULT_ZARR = os.path.join(PROJECT_ROOT, DEFAULT_ZARR_REL)



# Catàleg d'algoritmes disponibles
ALGORITHMS: Dict[str, Dict[str, Any]] = {
    "cardiac_output": {
        "required_tracks": CO_REQUIRED_TRACKS,
        "runner": CardiacOutput,  # es cridarà com CardiacOutput(zarr_path)
    },
    # Aquí hi podràs afegir més algoritmes en el futur:
    # "ppv": {"required_tracks": PPV_REQUIRED_TRACKS, "runner": PPV},
}
def _safe_group(root: zarr.hierarchy.Group, path: str) -> zarr.hierarchy.Group:
    """
    Crea (si cal) i retorna un grup Zarr donada una ruta tipus "algorithms/cardiac_output".
    """
    parts = [p for p in path.split("/") if p]
    grp = root
    for p in parts:
        if p in grp:
            grp = grp[p]
        else:
            grp = grp.create_group(p)
    return grp


def _append_1d(ds: zarr.core.Array, new_data: np.ndarray) -> None:
    """
    Afegeix new_data (1D) al final del dataset Zarr (append).
    """
    if new_data.size == 0:
        return
    old_len = ds.shape[0]
    ds.resize(old_len + new_data.size)
    ds[old_len:] = new_data


def append_algorithm_point(zarr_path: str, algo_name: str, time_ms: int, value: float) -> None:
    """
    Desa un punt nou (time_ms, value) per a un algoritme dins del .zarr, en:
        algorithms/<algo_name>/{time_ms, value}

    Preparat per streaming: cada crida afegeix una nova mostra.
    """
    root = zarr.open_group(zarr_path, mode="a")
    grp = _safe_group(root, f"algorithms/{algo_name}")

    # time_ms: int64, value: float32
    ds_time = grp.require_dataset(
        "time_ms",
        shape=(0,),
        chunks=(1024,),
        dtype="int64",
        compressor=None,  # opcionalment, podries usar el mateix compressor que als signals
    )
    ds_val = grp.require_dataset(
        "value",
        shape=(0,),
        chunks=(1024,),
        dtype="float32",
        compressor=None,
    )

    t_arr = np.asarray([time_ms], dtype="int64")
    v_arr = np.asarray([value], dtype="float32")

    _append_1d(ds_time, t_arr)
    _append_1d(ds_val, v_arr)

    grp.attrs.setdefault("algorithm", algo_name)


def prepare_zarr_for_algorithms(
    vital_path: str,
    zarr_path: str,
    algo_names: List[str],
    window_secs: float | None = None,
) -> None:
    """
    Calcula la unió de totes les REQUIRED_TRACKS dels algoritmes seleccionats
    i crida vital_to_zarr UNA sola vegada per exportar-les/actualitzar-les.
    """
    all_tracks: set[str] = set()

    for name in algo_names:
        info = ALGORITHMS.get(name)
        if info is None:
            print(f"[WARN] Algoritme desconegut: {name}")
            continue
        all_tracks.update(info["required_tracks"])

    if not all_tracks:
        print("[WARN] No hi ha tracks a exportar (cap algoritme vàlid).")
        return

    print("\n[DISPATCHER] Algoritmes seleccionats:", algo_names)
    print("[DISPATCHER] Tracks a exportar/actualitzar:")
    for t in sorted(all_tracks):
        print("   -", t)

    vital_to_zarr(
        vital_path=vital_path,
        zarr_path=zarr_path,
        tracks=sorted(all_tracks),
        window_secs=window_secs,
    )


def run_algorithms_on_zarr(zarr_path: str, algo_names: List[str]) -> Dict[str, Any]:
    """
    Executa els algoritmes seleccionats sobre el mateix .zarr.
    - Retorna un diccionari {nom_algoritme: instància/resultat}.
    - Desa també el resultat "instantani" de cada algoritme a:
        algorithms/<nom_algoritme>/{time_ms, value}
      perquè es pugui fer una gràfica retrospectiva (streaming-ready).
    """
    results: Dict[str, Any] = {}

    for name in algo_names:
        info = ALGORITHMS.get(name)
        if info is None:
            print(f"[WARN] Algoritme desconegut: {name}")
            continue

        runner = info["runner"]

        try:
            algo_instance = runner(zarr_path)  # p.ex. CardiacOutput(zarr_path)
            results[name] = algo_instance

            # Guardem al Zarr el punt resultant, si tenim el que cal
            if hasattr(algo_instance, "co_last") and hasattr(algo_instance, "t_last_ms") and name == "cardiac_output":
                append_algorithm_point(
                    zarr_path=zarr_path,
                    algo_name=name,
                    time_ms=algo_instance.t_last_ms,
                    value=algo_instance.co_last,
                )
                print(f"[ALG-STORE] {name}: guardat punt al Zarr (t_ms={algo_instance.t_last_ms}, value={algo_instance.co_last:.2f})")

            # En el futur, per altres algoritmes, podràs seguir el mateix patró:
            # if name == "ppv" and hasattr(algo_instance, "ppv_last") and hasattr(algo_instance, "t_last_ms"):
            #     append_algorithm_point(...)

        except ValueError as e:
            # Errors tipus "no hi ha SV", "totes les mostres són NaN", etc.
            print(f"[WARN] No s'ha pogut calcular '{name}': {e}")
            results[name] = None

    return results



if __name__ == "__main__":
    # 1) Fem servir les rutes per defecte
    vital_path = DEFAULT_VITAL
    zarr_path = DEFAULT_ZARR

   
    print(f"Fitxer .vital origen : {vital_path}")
    print(f"Fitxer .zarr sortida : {zarr_path}")

    # 2) Tria MANUAL dels algoritmes per terminal
    print("\nAlgoritmes disponibles:")
    for name in ALGORITHMS.keys():
        print(" -", name)

    raw = input("\nEscriu els algoritmes a executar (separats per comes): ")
    algo_names = [a.strip() for a in raw.split(",") if a.strip()]

    if not algo_names:
        print("[INFO] No s'ha seleccionat cap algoritme. Surto.")
        raise SystemExit(0)

    # 3) Assegurem que les tracks necessàries són al Zarr (idempotent)
    prepare_zarr_for_algorithms(vital_path, zarr_path, algo_names, window_secs=None)

    # 4) Executem els algoritmes seleccionats sobre el Zarr
    results = run_algorithms_on_zarr(zarr_path, algo_names)

print("\n========== RESULTATS ==========")
for name, instance in results.items():
    print(f"\n[{name}]")
    if instance is None:
        print("  (no disponible per aquest cas)")
    else:
        print(" ", instance)
print("================================")
