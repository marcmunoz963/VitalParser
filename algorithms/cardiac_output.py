import zarr
import numpy as np

# Config de tracks
SV_TRACK = "Intellivue/VOL_BLD_STROKE"
HR_CANDIDATES = [
    "Intellivue/ECG_HR",
    "Intellivue/ABP_HR",
    "Intellivue/HR",
]

# Llista total per al dispatcher
REQUIRED_TRACKS = [SV_TRACK] + HR_CANDIDATES


def _get_signals_group(root):
    if "signals" not in root:
        raise KeyError("El Zarr no conté el grup 'signals'")
    return root["signals"]


def _track_exists(signals, track_name: str) -> bool:
    if "/" not in track_name:
        return False
    vendor, track = track_name.split("/", 1)
    return vendor in signals and track in signals[vendor]


def _read_full_values_and_time(signals, track_name: str):
    """
    Retorna (ts_ms, vals) per un track:
      - ts_ms: np.ndarray int64 amb time_ms
      - vals:  np.ndarray float64 amb value
    """
    vendor, track = track_name.split("/", 1)
    grp = signals[vendor][track]

    if "value" not in grp or "time_ms" not in grp:
        raise KeyError(f"El track {track_name} no té 'value' i/o 'time_ms'")

    vals = grp["value"][:]  # float32
    ts_ms = grp["time_ms"][:]  # int64

    if vals.size == 0:
        raise ValueError(f"El track {track_name} no té dades")

    # Ens assegurem de treballar amb float64 per als càlculs
    return ts_ms.astype(np.int64), vals.astype(float)


class CardiacOutput:
    """
    Calcula el Cardiac Output com:
        CO = SV * HR

    - Utilitza l'ÚLTIMA mostra VÀLIDA (no NaN) on SV i HR existeixen.
    - Guarda també el timestamp corresponent (t_last_ms) per poder
      emmagatzemar el resultat en el Zarr i fer gràfiques històriques.
    """

    def __init__(self, zarr_path: str):
        # Obrim el Zarr i el grup de senyals
        root = zarr.open_group(zarr_path, mode="r")
        signals = _get_signals_group(root)

        # 1) Comprovem que existeix SV
        if not _track_exists(signals, SV_TRACK):
            raise ValueError(
                f"No s'ha trobat el track de Stroke Volume requerit: {SV_TRACK}"
            )

        # 2) Triem el primer HR candidat que existeixi
        hr_track = None
        for cand in HR_CANDIDATES:
            if _track_exists(signals, cand):
                hr_track = cand
                break

        if hr_track is None:
            raise ValueError(
                f"No s'ha trobat cap track de HR entre: {HR_CANDIDATES}"
            )

        # 3) Llegim TOTA la sèrie (temps + valors) de SV i HR
        ts_sv_ms, sv = _read_full_values_and_time(signals, SV_TRACK)
        ts_hr_ms, hr = _read_full_values_and_time(signals, hr_track)

        # Ens quedem amb la longitud comuna mínima
        n = min(len(sv), len(hr), len(ts_sv_ms), len(ts_hr_ms))
        if n == 0:
            raise ValueError("CardiacOutput: no hi ha dades suficients (0 mostres).")

        sv = sv[:n]
        hr = hr[:n]
        ts_sv_ms = ts_sv_ms[:n]
        ts_hr_ms = ts_hr_ms[:n]

        # 4) Filtrar mostres vàlides (no NaN a cap dels dos)
        valid_mask = ~np.isnan(sv) & ~np.isnan(hr)
        if not np.any(valid_mask):
            raise ValueError(
                "CardiacOutput: totes les mostres SV/HR són NaN (cap punt útil)."
            )

        # Índex de l'última mostra vàlida
        last_idx = int(np.nonzero(valid_mask)[0][-1])

        # Guardem sèries i informació útil per a futurs usos
        self.sv_series = sv
        self.hr_series = hr
        self.valid_mask = valid_mask
        self.hr_track_used = hr_track

        self.sv_last = float(sv[last_idx])
        self.hr_last = float(hr[last_idx])
        self.co_last = self.sv_last * self.hr_last

        # Timestamp corresponent a SV en aquesta mostra (ms)
        self.t_last_ms = int(ts_sv_ms[last_idx])
        # Per comoditat, també en segons
        self.t_last = self.t_last_ms / 1000.0

    def __repr__(self) -> str:
        return (
            f"CardiacOutput(co_last={self.co_last:.2f}, "
            f"sv_last={self.sv_last:.2f}, hr_last={self.hr_last:.2f}, "
            f"t_last_ms={self.t_last_ms}, "
            f"hr_track='{self.hr_track_used}')"
        )
