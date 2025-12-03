import os
import json
import random

import cv2
import numpy as np
import pandas as pd

# ============================================================
# CONFIG – update BASE_DIR if needed
# ============================================================
BASE_DIR      = r"C:\ECE253Project"

IMAGES_ROOT   = os.path.join(BASE_DIR, "100k")         # 100k/train, 100k/val, 100k/test
LABELS_ROOT   = os.path.join(BASE_DIR, "100k_labels")  # 100k_labels/train, ... (JSON)

OUT_ROOT      = os.path.join(BASE_DIR, "multi_degraded_100k")
os.makedirs(OUT_ROOT, exist_ok=True)

# how many synthetic samples per original image
N_SYNTHETIC_PER_IMAGE = 3

# motion blur ranges
L_MIN_SYN = 5.0
L_MAX_SYN = 60.0
THETA_MIN_SYN = 0.0
THETA_MAX_SYN = 180.0

# fog strength range
FOG_MIN = 0.3
FOG_MAX = 0.8

# low-light params range
LOWLIGHT_BRIGHT_MIN = 0.2
LOWLIGHT_BRIGHT_MAX = 0.5
LOWLIGHT_GAMMA_MIN = 1.2
LOWLIGHT_GAMMA_MAX = 1.8

# synthetic codes we may sample *per image*
# 1=blur, 2=fog, 3=lowlight
ALL_CODES = ["1", "2", "3", "12", "13", "23", "123"]

# weather labels we treat as "fog-like" (you can tune)
FOG_WEATHERS = {"foggy"}  # you can add "snowy", "rainy" if you want
# time-of-day labels we treat as low-light
LOWLIGHT_TOD = {"night", "dawn/dusk"}


# ============================================================
# BASIC DEGRADATION FUNCTIONS
# ============================================================

def motion_blur(img_rgb, L=15, theta=45):
    """Apply linear motion blur of length L and angle theta to an RGB uint8 image."""
    ksize = max(3, int(L))
    if ksize % 2 == 0:
        ksize += 1

    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0

    M = cv2.getRotationMatrix2D((ksize / 2 - 0.5, ksize / 2 - 0.5), theta, 1.0)
    kernel = cv2.warpAffine(
        kernel, M, (ksize, ksize),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    s = kernel.sum()
    if s > 0:
        kernel /= s

    blurred = cv2.filter2D(img_rgb, -1, kernel, borderType=cv2.BORDER_REFLECT)
    return blurred


def add_fog(img_rgb, strength=0.5):
    """Simple fog: blend towards light gray."""
    strength = float(np.clip(strength, 0.0, 1.0))
    img = img_rgb.astype(np.float32) / 255.0
    A = np.array([0.9, 0.9, 0.9], dtype=np.float32).reshape(1, 1, 3)

    fogged = (1.0 - strength) * img + strength * A
    fogged = np.clip(fogged * 255.0, 0, 255).astype(np.uint8)
    return fogged


def add_low_light(img_rgb, brightness=0.4, gamma=1.5):
    """Darken + gamma to simulate low-light."""
    img = img_rgb.astype(np.float32) / 255.0
    img = img * brightness
    img = np.power(img, gamma)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


# ============================================================
# HELPERS
# ============================================================

def read_attributes(json_path):
    """Read BDD100K attribute JSON and return (weather, timeofday)."""
    if not os.path.exists(json_path):
        return None, None

    with open(json_path, "r") as f:
        data = json.load(f)
    attrs = data.get("attributes", {})
    weather = attrs.get("weather", None)
    tod = attrs.get("timeofday", None)
    return weather, tod


def is_natural_fog(weather):
    return weather in FOG_WEATHERS if weather is not None else False


def is_natural_lowlight(tod):
    return tod in LOWLIGHT_TOD if tod is not None else False


# ============================================================
# MAIN DATASET BUILD
# ============================================================

def build_multitask_dataset():
    rows = []

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(IMAGES_ROOT, split)
        label_dir = os.path.join(LABELS_ROOT, split)
        out_split_dir = os.path.join(OUT_ROOT, split)
        os.makedirs(out_split_dir, exist_ok=True)

        print(f"\n=== Processing split: {split} ===")
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]
        img_files.sort()

        for idx, fname in enumerate(img_files, start=1):
            img_path = os.path.join(img_dir, fname)
            base, _ = os.path.splitext(fname)
            json_path = os.path.join(label_dir, base + ".json")

            # read image
            bgr = cv2.imread(img_path)
            if bgr is None:
                print(f"  [WARN] could not read image: {img_path}")
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # read attributes
            weather, tod = read_attributes(json_path)
            nat_fog = is_natural_fog(weather)
            nat_low = is_natural_lowlight(tod)

            # ------------- 1) CLEAN SAMPLE -------------
            clean_name = f"CLEAN_{fname}"
            clean_out_path = os.path.join(out_split_dir, clean_name)
            # save as BGR for OpenCV
            cv2.imwrite(clean_out_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            rows.append({
                "set": split,
                "filepath": clean_out_path,
                "blur_present": 0,
                "fog_present": 1 if nat_fog else 0,
                "lowlight_present": 1 if nat_low else 0,
                "L": 0.0,
                "theta": 0.0,
                "natural_fog": int(nat_fog),
                "natural_lowlight": int(nat_low),
                "synthetic_code": "0"
            })

            # ------------- 2) SYNTHETIC SAMPLES ----------
            # Don't add synthetic fog on natural fog,
            # don't add synthetic low-light on natural low-light.
            candidate_codes = [
                code for code in ALL_CODES
                if not (nat_fog and "2" in code)
                and not (nat_low and "3" in code)
            ]
            if len(candidate_codes) == 0:
                # fall back to just blur if everything was filtered out
                candidate_codes = ["1"]

            for s_idx in range(N_SYNTHETIC_PER_IMAGE):
                code = random.choice(candidate_codes)
                deg = rgb.copy()

                blur_present = 0
                fog_present = 1 if nat_fog else 0  # start with natural state
                low_present = 1 if nat_low else 0

                L = 0.0
                theta = 0.0

                # synthetic blur
                if "1" in code:
                    L = float(np.random.uniform(L_MIN_SYN, L_MAX_SYN))
                    theta = float(np.random.uniform(THETA_MIN_SYN, THETA_MAX_SYN))
                    deg = motion_blur(deg, L=L, theta=theta)
                    blur_present = 1

                # synthetic fog
                if "2" in code:
                    fs = float(np.random.uniform(FOG_MIN, FOG_MAX))
                    deg = add_fog(deg, strength=fs)
                    fog_present = 1

                # synthetic low-light
                if "3" in code:
                    b = float(np.random.uniform(LOWLIGHT_BRIGHT_MIN, LOWLIGHT_BRIGHT_MAX))
                    g = float(np.random.uniform(LOWLIGHT_GAMMA_MIN, LOWLIGHT_GAMMA_MAX))
                    deg = add_low_light(deg, brightness=b, gamma=g)
                    low_present = 1

                out_name = f"DEG{s_idx+1}_{code}_{fname}"
                out_path = os.path.join(out_split_dir, out_name)
                cv2.imwrite(out_path, cv2.cvtColor(deg, cv2.COLOR_RGB2BGR))

                rows.append({
                    "set": split,
                    "filepath": out_path,
                    "blur_present": blur_present,
                    "fog_present": fog_present,
                    "lowlight_present": low_present,
                    "L": L if blur_present else 0.0,
                    "theta": theta if blur_present else 0.0,
                    "natural_fog": int(nat_fog),
                    "natural_lowlight": int(nat_low),
                    "synthetic_code": code
                })

            if idx % 500 == 0:
                print(f"  Processed {idx}/{len(img_files)} images in {split}...")

    # ------------- WRITE CSV -------------
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_ROOT, "labels_multitask.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Done! Wrote {len(df)} rows to {csv_path}")


if __name__ == "__main__":
    build_multitask_dataset()
