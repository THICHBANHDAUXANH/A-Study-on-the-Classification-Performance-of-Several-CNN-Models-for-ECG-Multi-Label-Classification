from __future__ import annotations

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from config import ARRAY_IDS_PATH, ARRAY_X_PATH, ARRAY_Y_PATH, IMAGE_DIR, IMAGE_SIZE, LABEL_DIR

def discover_records() -> list[tuple[int, object, object]]:
    image_paths = {int(path.stem): path for path in sorted(IMAGE_DIR.glob("*.png")) if path.stem.isdigit()}
    label_paths = {int(path.stem): path for path in sorted(LABEL_DIR.glob("*.npy")) if path.stem.isdigit()}
    common_ids = sorted(set(image_paths) & set(label_paths))
    if not common_ids:
        raise RuntimeError(f"No paired image/label files found in {IMAGE_DIR} and {LABEL_DIR}")
    return [(ecg_id, image_paths[ecg_id], label_paths[ecg_id]) for ecg_id in common_ids]

def load_image(image_path) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        return np.asarray(image, dtype=np.uint8)

def main() -> None:
    records = discover_records()
    images = []
    labels = []
    ids = []

    for ecg_id, image_path, label_path in tqdm(records, desc="Building numpy arrays", unit="record"):
        images.append(load_image(image_path))
        labels.append(np.load(label_path, allow_pickle=False).astype(np.float32))
        ids.append(ecg_id)

    ARRAY_X_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(ARRAY_X_PATH, np.stack(images).astype(np.uint8))
    np.save(ARRAY_Y_PATH, np.stack(labels).astype(np.float32))
    np.save(ARRAY_IDS_PATH, np.asarray(ids, dtype=np.int32))
    print(f"Saved X: {ARRAY_X_PATH}")
    print(f"Saved y: {ARRAY_Y_PATH}")
    print(f"Saved ids: {ARRAY_IDS_PATH}")

if __name__ == "__main__":
    main()
