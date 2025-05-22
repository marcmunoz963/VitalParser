import os
import math

# Theme colors
DARK_BG = "#2e2e2e"
LIGHT_BG = "#3d3e3e"
FG_COLOR = "#ffffff"
ACCENT = "#4d90fe"


def is_nan(x):
    try:
        return math.isnan(x)
    except Exception:
        return False


def key_datetime(fname):
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split('_')
    return parts[-2] + parts[-1]


def find_latest_vital(recordings_dir):
    if not os.path.isdir(recordings_dir):
        return None

    # Numeric subfolders
    folders = [d for d in os.listdir(recordings_dir)
               if os.path.isdir(os.path.join(recordings_dir, d)) and d.isdigit()]
    if not folders:
        return None

    latest_folder = sorted(folders)[-1]
    folder_path = os.path.join(recordings_dir, latest_folder)

    # Latest .vital file
    vitals = [f for f in os.listdir(folder_path) if f.endswith('.vital')]
    if not vitals:
        return None

    latest = sorted(vitals, key=key_datetime)[-1]
    return os.path.join(folder_path, latest)

