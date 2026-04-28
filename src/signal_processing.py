from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sg
import wfdb
from sklearn.preprocessing import MultiLabelBinarizer

from config import PTBXL_META_CSV, PTBXL_RECORD_BASE, PTBXL_SCP_CSV, REPORT_CLASSES

def smooth_signal(ecg: np.ndarray, window_size: int = 5) -> np.ndarray:
    kernel = np.ones(window_size, dtype=float) / window_size
    return sg.convolve(ecg, kernel, mode="same")

def notch_filter(ecg: np.ndarray, fs: int = 100, freq: float = 50.0, q_factor: float = 30.0) -> np.ndarray:
    notch_freq = min(freq, fs / 2 - 1e-3)
    b, a = sg.iirnotch(notch_freq, q_factor, fs)
    return sg.filtfilt(b, a, ecg, axis=0)

def highpass_filter(ecg: np.ndarray, fs: int = 100, cutoff: float = 0.5, order: int = 3) -> np.ndarray:
    b, a = sg.butter(order, cutoff / (0.5 * fs), btype="high")
    return sg.filtfilt(b, a, ecg, axis=0)

def lowpass_filter(ecg: np.ndarray, fs: int = 100, cutoff: float = 0.5, order: int = 3) -> np.ndarray:
    b, a = sg.butter(order, cutoff / (0.5 * fs), btype="low")
    return sg.filtfilt(b, a, ecg, axis=0)

def preprocess_record(record: np.ndarray, fs: int = 100) -> np.ndarray:
    smoothed = smooth_signal(record, window_size=5)
    notched = notch_filter(smoothed, fs=fs, freq=50.0)
    return highpass_filter(notched, fs=fs, cutoff=0.5, order=3)

def compute_psd(signal_1d: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(1024, signal_1d.size)
    return sg.welch(signal_1d, fs=fs, nperseg=nperseg)

def bandpower(signal_1d: np.ndarray, fs: int, low: float, high: float) -> float:
    freqs, psd = compute_psd(signal_1d, fs)
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))

def load_metadata() -> pd.DataFrame:
    meta = pd.read_csv(PTBXL_META_CSV)
    meta = meta[meta["filename_lr"].notna() & meta["ecg_id"].notna()].copy()
    meta["ecg_id"] = meta["ecg_id"].astype(int)
    meta["scp_codes"] = meta["scp_codes"].apply(ast.literal_eval)
    return meta.reset_index(drop=True)

def load_scp_superclass_map() -> pd.DataFrame:
    statements = pd.read_csv(PTBXL_SCP_CSV, index_col=0)
    return statements[statements["diagnostic"] == 1]

def aggregate_diagnostic_superclass(scp_code_dict: dict[str, float], superclass_map: pd.DataFrame) -> list[str]:
    labels = []
    for code in scp_code_dict:
        if code in superclass_map.index:
            labels.append(str(superclass_map.loc[code, "diagnostic_class"]))
    return [label for label in REPORT_CLASSES if label in set(labels)]

def attach_multilabel_targets(meta: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    superclass_map = load_scp_superclass_map()
    records = meta.copy()
    records["superclass"] = records["scp_codes"].apply(
        lambda value: aggregate_diagnostic_superclass(value, superclass_map)
    )
    records = records[records["superclass"].map(len) > 0].reset_index(drop=True)
    mlb = MultiLabelBinarizer(classes=REPORT_CLASSES)
    labels = mlb.fit_transform(records["superclass"]).astype(np.float32)
    return records, labels

def load_wfdb_record(filename_lr: str) -> tuple[np.ndarray, int]:
    record_path = PTBXL_RECORD_BASE/Path(filename_lr).with_suffix("")
    signal, fields = wfdb.rdsamp(str(record_path))
    fs = int(fields.get("fs", 100))
    return signal, fs

def save_ecg_image(signal: np.ndarray, output_path: Path) -> None:
    fig, axes = plt.subplots(12, 1, figsize=(10, 10), sharex=True)
    for lead_idx, ax in enumerate(axes):
        ax.plot(signal[:, lead_idx], linewidth=0.8)
        ax.set_ylabel(f"Lead {lead_idx + 1}")
        ax.set_yticks([])
        ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
