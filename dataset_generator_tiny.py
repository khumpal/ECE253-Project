import os
import json
import random
import cv2
import numpy as np
import pandas as pd

BASE_DIR      = r"C:\ECE253Project"
IMAGES_ROOT   = os.path.join(BASE_DIR, "100k")
LABELS_ROOT   = os.path.join(BASE_DIR, "100k_labels")

OUT_ROOT      = os.path.join(BASE_DIR, "multi_degraded_tiny")
os.makedirs(OUT_ROOT, exist_ok=True)

MAX_IMAGES_PER_SPLIT = 5
N_SYNTHETIC_PER_IMAGE = 1

L_MIN_SYN = 5.0
L_MAX_SYN = 60.0
THETA_MIN_SYN = 0.0
THETA_MAX_SYN = 180.0

FOG_MIN = 0.3
FOG_MAX = 0.8
LOWLIGHT_BRIGHT_MIN = 0.2
LOWLIGHT_BRIGHT_MAX = 0.5
LOWLIGHT_GAMMA_MIN = 1.2
LOWLIGHT_GAMMA_MAX = 1.8

ALL_CODES = ["1", "2", "3", "12", "13", "23", "123"]

FOG_WEATHERS = {"foggy"}
LOWLIGHT_TOD = {"night", "dawn/dusk"}


def motion_blur(img_rgb, L=15, theta=45):
    ksize = max(3, int(L))
    if ksize % 2 == 0:
        ksize += 1
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((ksize/2 - 0.5, ksize/2 - 0.5), theta, 1)
    kernel = cv2.warpAffine(kernel, M, (ksize, ksize),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
    kernel /= kernel.sum()
    return cv2.filter2D(img_rgb, -1, kernel, borderType=cv2.BORDER_REFLECT)


def add_fog(img_rgb, strength=0.5):
    img = img_rgb.astype(np.float32) / 255.0
    A = np.array([0.9, 0.9, 0.9], dtype=np.float32).reshape(1, 1, 3)
    fogged = (1 - strength) * img + strength * A
    return np.clip(fogged * 255, 0, 255).astype(np.uint8)


def add_low_light(img_rgb, brightness=0.4, gamma=1.5):
    img = (img_rgb.astype(np.float32) / 255.0) * brightness
    img = np.power(img, gamma)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def read_attributes(json_path):
    if not os.path.exists(json_path):
        return None, None
    with open(json_path, "r") as f:
        data = json.load(f)
    attrs = data.get("attributes", {})
    return attrs.get("weather", None), attrs.get("timeofday", None)


def is_natural_fog(weather):
    return weather in FOG_WEATHERS if weather else False


def is_natural_lowlight(tod):
    return tod in LOWLIGHT_TOD if tod else False


def build_multitask_dataset():
    rows = []

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(IMAGES_ROOT, split)
        label_dir = os.path.join(LABELS_ROOT, split)
        out_split_dir = os.path.join(OUT_ROOT, split)
        os.makedirs(out_split_dir, exist_ok=True)

        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]
        img_files.sort()
        img_files = img_files[:MAX_IMAGES_PER_SPLIT]

        for fname in img_files:
            img_path = os.path.join(img_dir, fname)
            base, _ = os.path.splitext(fname)
            json_path = os.path.join(label_dir, base + ".json")

            bgr = cv2.imread(img_path)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            weather, tod = read_attributes(json_path)
            nat_fog = is_natural_fog(weather)
            nat_low = is_natural_lowlight(tod)

            clean_name = f"CLEAN_{fname}"
            clean_out_path = os.path.join(out_split_dir, clean_name)
            cv2.imwrite(clean_out_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            rows.append({
                "set": split,
                "filepath": clean_out_path,
                "blur_present": 0,
                "fog_present": int(nat_fog),
                "lowlight_present": int(nat_low),
                "L": 0.0,
                "theta": 0.0,
                "natural_fog": int(nat_fog),
                "natural_lowlight": int(nat_low),
                "synthetic_code": "0"
            })

            candidate_codes = [
                code for code in ALL_CODES
                if not (nat_fog and "2" in code)
                and not (nat_low and "3" in code)
            ]
            if not candidate_codes:
                candidate_codes = ["1"]

            code = random.choice(candidate_codes)
            deg = rgb.copy()

            blur_present = 0
            fog_present = int(nat_fog)
            low_present = int(nat_low)

            L = 0.0
            theta = 0.0

            if "1" in code:
                L = float(np.random.uniform(L_MIN_SYN, L_MAX_SYN))
                theta = float(np.random.uniform(THETA_MIN_SYN, THETA_MAX_SYN))
                deg = motion_blur(deg, L, theta)
                blur_present = 1

            if "2" in code:
                fs = float(np.random.uniform(FOG_MIN, FOG_MAX))
                deg = add_fog(deg, fs)
                fog_present = 1

            if "3" in code:
                b = float(np.random.uniform(LOWLIGHT_BRIGHT_MIN, LOWLIGHT_BRIGHT_MAX))
                g = float(np.random.uniform(LOWLIGHT_GAMMA_MIN, LOWLIGHT_GAMMA_MAX))
                deg = add_low_light(deg, b, g)
                low_present = 1

            out_name = f"DEG_{code}_{fname}"
            out_path = os.path.join(out_split_dir, out_name)
            cv2.imwrite(out_path, cv2.cvtColor(deg, cv2.COLOR_RGB2BGR))

            rows.append({
                "set": split,
                "filepath": out_path,
                "blur_present": blur_present,
                "fog_present": fog_present,
                "lowlight_present": low_present,
                "L": L,
                "theta": theta,
                "natural_fog": int(nat_fog),
                "natural_lowlight": int(nat_low),
                "synthetic_code": code
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_ROOT, "labels_multitask_tiny.csv")
    df.to_csv(csv_path, index=False)
    print("Done. Wrote:", csv_path)


if __name__ == "__main__":
    build_multitask_dataset()
