from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import IMAGE_DIR, LABEL_DIR, PREPROCESSED_SIGNAL_DIR, TABLE_DIR
from signal_processing import attach_multilabel_targets, load_metadata, load_wfdb_record, preprocess_record, save_ecg_image

MAX_RECORDS = 0
SKIP_EXISTING = True

def main() -> None:
    PREPROCESSED_SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    records, labels = attach_multilabel_targets(load_metadata())
    if MAX_RECORDS > 0:
        records = records.head(MAX_RECORDS).reset_index(drop=True)
        labels = labels[:MAX_RECORDS]

    index_rows = []
    failed_rows = []
    iterator = tqdm(records.itertuples(index=True), total=len(records), desc="Preprocessing records", unit="record")
    for row in iterator:
        idx = int(row.Index)
        ecg_id = int(row.ecg_id)
        signal_path = PREPROCESSED_SIGNAL_DIR/f"{ecg_id}.npy"
        image_path = IMAGE_DIR/f"{ecg_id}.png"
        label_path = LABEL_DIR/f"{ecg_id}.npy"

        try:
            if not (SKIP_EXISTING and signal_path.exists() and image_path.exists() and label_path.exists()):
                raw_signal, fs = load_wfdb_record(str(row.filename_lr))
                processed_signal = preprocess_record(raw_signal, fs=fs)
                np.save(signal_path, processed_signal.astype(np.float32))
                save_ecg_image(processed_signal, image_path)
                np.save(label_path, labels[idx])

            index_rows.append(
                {
                    "ecg_id": ecg_id,
                    "filename_lr": row.filename_lr,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "signal_path": str(signal_path),
                    "superclass": "|".join(row.superclass),
                }
            )
        except Exception as exc:
            failed_rows.append({"ecg_id": ecg_id, "error": f"{type(exc).__name__}: {exc}"})

    pd.DataFrame(index_rows).to_csv(TABLE_DIR/"dataset_index.csv", index=False)
    pd.DataFrame(failed_rows).to_csv(TABLE_DIR/"preprocessing_failures.csv", index=False)
    print(f"Completed preprocessing. Success: {len(index_rows)} / {len(records)}")
    print(f"Dataset index: {TABLE_DIR/'dataset_index.csv'}")

if __name__ == "__main__":
    main()
