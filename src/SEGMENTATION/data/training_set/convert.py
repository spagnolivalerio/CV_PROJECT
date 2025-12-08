import os
import json
import numpy as np
import cv2

META_PATH = "meta.json"
ANN_DIR   = "d2/ann"
OUT_MASKS = "../masks"

os.makedirs(OUT_MASKS, exist_ok=True)


with open(META_PATH, "r") as f:
    meta = json.load(f)

classid_to_index = {}
idx = 1  # 0 = background

for c in meta["classes"]:
    classid_to_index[c["id"]] = idx
    idx += 1

print("ClassID → Index mapping:")
print(classid_to_index)


for fname in os.listdir(ANN_DIR):
    if not fname.endswith(".json"):
        continue

    json_path = os.path.join(ANN_DIR, fname)
    with open(json_path, "r") as f:
        ann = json.load(f)

    H = ann["size"]["height"]
    W = ann["size"]["width"]

    mask = np.zeros((H, W), dtype=np.uint8)

    for obj in ann["objects"]:
        class_id = obj["classId"]

        if class_id not in classid_to_index:
            continue

        cls_index = classid_to_index[class_id]

        points = np.array(obj["points"]["exterior"], dtype=np.int32)

        cv2.fillPoly(mask, [points], color=cls_index)

    out_name = fname.replace(".json", ".png")
    out_path = os.path.join(OUT_MASKS, out_name)

    cv2.imwrite(out_path, mask)

    print(f"Saved: {out_path}")

