import zarr
import os
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional

STORE_PATH = os.path.join("results", "BOX01_20251024.zarr")

def open_root(store_path=STORE_PATH):
    return zarr.open(store_path, mode="r")

def epoch1700_to_datetime(ts_seconds: float) -> datetime:
    """Converteix segons des de 1700-01-01 UTC a datetime ISO."""
    epoch_1700 = datetime(1700, 1, 1, tzinfo=timezone.utc)
    return epoch_1700 + timedelta(seconds=float(ts_seconds))

def load_track(root, track_path: str):
    """
    Retorna (t_abs_ms, t_rel_ms, vals) per a una track.
    t_abs_ms: np.ndarray[int64] timestamps absoluts en ms (origen 1700)
    t_rel_ms: np.ndarray[int64] temps relatiu en ms, t_rel_ms[0] == 0
    vals:     np.ndarray valors de la senyal
    """
    t_key = f"{track_path}_time_ms"
    if t_key not in root:
        raise KeyError(f"No s'ha trobat {t_key} dins del Zarr.")
    if track_path not in root:
        raise KeyError(f"No s'ha trobat {track_path} dins del Zarr.")

    t_abs_ms = root[t_key][:].astype(np.int64)
    vals = root[track_path][:]

    if t_abs_ms.size == 0:
        return t_abs_ms, t_abs_ms, vals

    t0 = int(t_abs_ms[0])
    t_rel_ms = (t_abs_ms - t0).astype(np.int64)
    return t_abs_ms, t_rel_ms, vals

def slice_by_seconds(t_rel_ms: np.ndarray, vals: np.ndarray, start_s: float, end_s: float):
    """
    Retorna (t_rel_ms_win, vals_win) per la finestra [start_s, end_s).
    """
    start_ms = int(start_s * 1000)
    end_ms = int(end_s * 1000)
    # Índexos via cerca binària
    i0 = np.searchsorted(t_rel_ms, start_ms, side="left")
    i1 = np.searchsorted(t_rel_ms, end_ms, side="left")
    return t_rel_ms[i0:i1], vals[i0:i1]

def dump_track(root, track_path: str, *, head: int = 10, tail: int = 10) -> None:
    """Imprimeix informació útil d'una track: mides, mostres, estadístiques bàsiques.
    """
    print(f"\n[TRACK] {track_path}")
    try:
        t_abs_ms, t_rel_ms, vals = load_track(root, track_path)
    except KeyError as e:
        print(f"  - No trobada: {e}")
        return

    n = vals.shape[0]
    print(f"  - Num mostres: {n}")
    if n == 0:
        print("  - (Buit)")
        return

    # Estatístiques ràpides
    finite_mask = np.isfinite(vals)
    n_finite = int(finite_mask.sum())
    print(f"  - Finite: {n_finite} / {n}")
    if n_finite > 0:
        finite_vals = vals[finite_mask]
        print(f"  - min={finite_vals.min():.6g}  max={finite_vals.max():.6g}  mean={finite_vals.mean():.6g}")

    # Mostrar cap i cua
    h = min(head, n)
    t = min(tail, n)
    print("  - Primeres mostres:")
    for ms, v in zip(t_rel_ms[:h], vals[:h]):
        print(f"    t={int(ms)} ms  v={v}")
    if n > h:
        print("  - Darreres mostres:")
        for ms, v in zip(t_rel_ms[-t:], vals[-t:]):
            print(f"    t={int(ms)} ms  v={v}")

# --- Navegació dinàmica de contingut (senyals/prediccions) ---

def _walk_arrays(node, base=""):
    """
    Retorna una llista de rutes completes a ARRAYS (no grups) sota 'node'.
    Cada element és una string com "signals/Intellivue/PLETH".
    """
    out = []
    try:
        items = list(node.items())
    except Exception:
        items = []
    for name, child in items:
        path = f"{base}/{name}" if base else name
        # Heurística: un array té "shape" i "dtype"
        if hasattr(child, "shape") and hasattr(child, "dtype"):
            out.append(path)
        else:
            # Suposem que és un grup; recórrer recursivament
            out.extend(_walk_arrays(child, base=path))
    return out

def list_available_tracks(root):
    """
    Llista arrays disponibles sota 'signals' i 'pred' (si existeixen).
    Retorna (signals, preds), dues llistes de rutes.
    """
    signals = _walk_arrays(root["signals"], base="signals") if "signals" in root else []
    preds   = _walk_arrays(root["pred"],     base="pred")     if "pred"     in root else []
    # filtrar els arrays _time_ms (no són valors)
    signals = [p for p in signals if not p.endswith("_time_ms")]
    preds   = [p for p in preds   if not p.endswith("_time_ms")]
    return signals, preds

if __name__ == "__main__":
    root = open_root()
    # Mostrem l'arbre per verificar què hi ha
    print(root.tree())

    # Tracks de senyals que esperem trobar (ajusta segons el que s'hagi escrit)
    pleth_path = "signals/Intellivue/PLETH"
    art_path   = "signals/Intellivue/ART"

    # Bolcat informatiu de cada track
    dump_track(root, pleth_path)
    dump_track(root, art_path)

    # Si hi ha prediccions, llista-les i mostra'n una
    print("\n[PREDICCIONS DISPONIBLES]")
    pred_group = "pred"
    if pred_group in root:
        # Llistar claus sota pred
        try:
            pred_keys = sorted([k for k in root[pred_group].keys()])
        except Exception:
            pred_keys = []
        for k in pred_keys:
            full = f"{pred_group}/{k}"
            node = root[full]
            kind = "Array" if hasattr(node, "shape") else "Node"
            shape = getattr(node, "shape", None)
            print(f"  - {full}  ({kind}{'' if shape is None else f' shape={shape}'})")
        # Mostra una predicció concreta si existeix
        for cand in ("BP_Prediction", "HTI", "HR_Pred", "Resp_Pred"):
            ppath = f"{pred_group}/{cand}"
            if ppath in root:
                t_key = f"{ppath}_time_ms"
                print(f"\n[DEMO] Mostrant {ppath}:")
                if t_key in root:
                    t = root[t_key][:]
                else:
                    t = None
                v = root[ppath][:]
                print(f"  shape={v.shape}")
                if v.size:
                    print("  Primeres 10 mostres:")
                    for i in range(min(10, v.size)):
                        ti = int(t[i]) if t is not None else None
                        print(f"    t={ti}  v={v[i]}")
                break
    else:
        print("  (No s'ha trobat cap grup 'pred')")

    # Exemple de finestra: PLETH del segon 300 al 360 (si existeix)
    try:
        t_abs, t_rel, vals = load_track(root, pleth_path)
        if t_rel.size:
            win_t, win_v = slice_by_seconds(t_rel, vals, start_s=300.0, end_s=360.0)
            print(f"\n[FINESTRA] PLETH [300s–360s): {win_v.shape[0]} mostres")
            
    except KeyError:
        pass

    # --- Versió 2: navegador interactiu de senyals/prediccions i finestra temporal ---
    try:
        print("\n[INTERACTIU] Selecció de font i finestra temporal")
        sigs, preds = list_available_tracks(root)

        if not sigs and not preds:
            print("  (No s'han trobat arrays sota 'signals' ni 'pred'.)")
        else:
            print("\n  Fonts disponibles:")
            if sigs:
                print("   [S] Senyals (signals/…):")
                for i, p in enumerate(sigs, 1):
                    print(f"     {i:2d}. {p}")
            else:
                print("   [S] Senyals: (cap)")

            if preds:
                print("   [P] Prediccions (pred/…):")
                for i, p in enumerate(preds, 1):
                    print(f"     {i:2d}. {p}")
            else:
                print("   [P] Prediccions: (cap)")

            # Triar col·lecció
            grp_choice = input("\n[INPUT] Tria col·lecció [S=signals, P=pred, ENTER per sortir]: ").strip().upper()
            if grp_choice in ("S", "P"):
                pool = sigs if grp_choice == "S" else preds
                if not pool:
                    print("  (No hi ha elements en aquesta col·lecció.)")
                else:
                    # Triar índex
                    while True:
                        idx_str = input(f"[INPUT] Introdueix l'índex (1..{len(pool)}): ").strip()
                        if not idx_str:
                            print("  (Cancel·lat.)")
                            break
                        try:
                            idx = int(idx_str)
                            if 1 <= idx <= len(pool):
                                sel_path = pool[idx - 1]
                                break
                            else:
                                print("  Índex fora de rang.")
                        except ValueError:
                            print("  Introdueix un enter vàlid.")
                    if sel_path:
                        # Llegir dades
                        try:
                            t_abs, t_rel, vals = load_track(root, sel_path)
                        except KeyError as e:
                            print(f"  No s'ha pogut carregar {sel_path}: {e}")
                            t_rel = np.array([], dtype=np.int64)
                            vals = np.array([], dtype=np.float32)

                        # Finestra temporal (ms o s)
                        unit = input("[INPUT] Unitat [ms/s] (ENTER=ms): ").strip().lower() or "ms"
                        start_str = input(f"[INPUT] Inici finestra ({unit}) (ENTER per ometre): ").strip()
                        end_str   = input(f"[INPUT] Final finestra ({unit}) (ENTER per ometre): ").strip()

                        if start_str and end_str and t_rel.size:
                            try:
                                if unit == "s":
                                    start_s = float(start_str)
                                    end_s   = float(end_str)
                                else:
                                    # ms → s
                                    start_s = float(start_str) / 1000.0
                                    end_s   = float(end_str)   / 1000.0
                                win_t, win_v = slice_by_seconds(t_rel, vals, start_s=start_s, end_s=end_s)
                                print(f"\n[RESULTAT] {sel_path} [{start_s:.3f}s–{end_s:.3f}s): {win_v.shape[0]} mostres")
                                if win_v.size:
                                    print("  Mostres dins de la finestra (t en ms):")
                                    for ms, v in zip(win_t, win_v):
                                        print(f"    t={int(ms)} ms  v={v}")
                            except Exception as e:
                                print(f"  No s'ha pogut aplicar la finestra: {e}")
                        else:
                            # Sense finestra: fer un resum ràpid
                            print(f"\n[RESUM] {sel_path}: {vals.shape[0]} mostres totals")
                            if vals.size:
                                finite = np.isfinite(vals)
                                if finite.any():
                                    fv = vals[finite]
                                    print(f"  min={fv.min():.6g}  max={fv.max():.6g}  mean={fv.mean():.6g}")
                                print("  Primeres 10 mostres:")
                                for ms, v in zip(t_rel[:10], vals[:10]):
                                    print(f"    t={int(ms)} ms  v={v}")
            else:
                print("  (Sortint de l'interactiu.)")
    except Exception as e:
        print(f"[WARN] No s'ha pogut executar el mode interactiu avançat: {e}")