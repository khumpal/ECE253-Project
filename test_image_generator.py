import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================
# FIXED IMAGE PATHS
# ==============================================================
CLEAN_IMAGE_PATH     = r"C:\ECE253Project\test_image.jpg"          # GT candidate
DISTORTED_IMAGE_PATH = r"C:\ECE253Project\test_image_distorted.jpg"
NATURAL_IMAGE_PATH   = r"C:\ECE253Project\test_image.jpg"          # same or different file


# ==============================================================
# DEGRADATION FUNCTIONS
# ==============================================================

def motion_blur(img_rgb, L=15, theta=45):
    ksize = max(3, int(L))
    if ksize % 2 == 0:
        ksize += 1

    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0

    M = cv2.getRotationMatrix2D((ksize / 2 - 0.5, ksize / 2 - 0.5), theta, 1.0)
    kernel = cv2.warpAffine(kernel, M, (ksize, ksize),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)

    s = kernel.sum()
    if s > 0:
        kernel /= s

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    blurred_bgr = cv2.filter2D(img_bgr, -1, kernel, borderType=cv2.BORDER_REFLECT)
    blurred_rgb = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2RGB)
    return blurred_rgb


def add_fog(img, strength=0.5):
    img = img.astype(np.float32) / 255.0
    A = np.array([0.9, 0.9, 0.9], dtype=np.float32)
    fogged = (1 - strength) * img + strength * A
    return np.clip(fogged * 255.0, 0, 255).astype(np.uint8)


def add_low_light(img, brightness=0.4, gamma=1.5):
    img = img.astype(np.float32) / 255.0
    img = (img * brightness) ** gamma
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


# ==============================================================
# CREATE TEST SAMPLE (FUNCTION FOR PIPELINE)
# ==============================================================

def create_test_sample(mode: str, degrade_code: str):
    """
    Create a test sample and SAVE the degraded image to DISTORTED_IMAGE_PATH.

    Args:
        mode: 's' for synthetic, 'n' for natural
        degrade_code: '0','1','2','3','12','23','123' (only used in synthetic)

    Returns:
        gt_rgb:      clean GT image (RGB uint8) or None (for natural)
        degraded_rgb: distorted image (RGB uint8)
        meta:       dict with degradation info
    """

    # ---- NATURAL IMAGE MODE ---------
    if mode == "n":
        img = cv2.imread(NATURAL_IMAGE_PATH)
        if img is None:
            raise FileNotFoundError(f"Could not read NATURAL_IMAGE_PATH: {NATURAL_IMAGE_PATH}")
        degraded_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Save distorted image for blur module
        os.makedirs(os.path.dirname(DISTORTED_IMAGE_PATH), exist_ok=True)
        cv2.imwrite(DISTORTED_IMAGE_PATH, cv2.cvtColor(degraded_rgb, cv2.COLOR_RGB2BGR))

        meta = {
            "mode": "natural",
            "code": "NA",
            "blur": False,
            "fog": False,
            "lowlight": False
        }
        return None, degraded_rgb, meta

    # ---- SYNTHETIC MODE -------------
    img = cv2.imread(CLEAN_IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Could not read CLEAN_IMAGE_PATH: {CLEAN_IMAGE_PATH}")

    gt_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    degraded_rgb = gt_rgb.copy()

    meta = {
        "mode": "synthetic",
        "code": degrade_code,
        "blur": False,
        "L": None,
        "theta": None,
        "fog": False,
        "fog_strength": None,
        "lowlight": False,
        "brightness": None,
        "gamma": None
    }

    if degrade_code != "0":
        # Blur
        if "1" in degrade_code:
            L = float(np.random.uniform(5.0, 60.0))
            T = float(np.random.uniform(0.0, 180.0))
            degraded_rgb = motion_blur(degraded_rgb, L=L, theta=T)
            meta["blur"] = True
            meta["L"] = L
            meta["theta"] = T

        # Fog
        if "2" in degrade_code:
            S = float(np.random.uniform(0.3, 0.8))
            degraded_rgb = add_fog(degraded_rgb, strength=S)
            meta["fog"] = True
            meta["fog_strength"] = S

        # Low-light
        if "3" in degrade_code:
            B = float(np.random.uniform(0.2, 0.5))
            G = float(np.random.uniform(1.2, 1.8))
            degraded_rgb = add_low_light(degraded_rgb, brightness=B, gamma=G)
            meta["lowlight"] = True
            meta["brightness"] = B
            meta["gamma"] = G

    # Save distorted image for blur module
    os.makedirs(os.path.dirname(DISTORTED_IMAGE_PATH), exist_ok=True)
    cv2.imwrite(DISTORTED_IMAGE_PATH, cv2.cvtColor(degraded_rgb, cv2.COLOR_RGB2BGR))

    return gt_rgb, degraded_rgb, meta


# ==============================================================
# OPTIONAL CLI DEMO
# ==============================================================

if __name__ == "__main__":
    print("\n=== Test Image Creator ===")
    print("Clean image path   :", CLEAN_IMAGE_PATH)
    print("Distorted will be saved as:", DISTORTED_IMAGE_PATH, "\n")

    mode = input("Synthetic(s) or Natural(n)? ").lower().strip()
    if mode not in ["s", "n"]:
        print("Invalid choice, defaulting to 's'.")
        mode = "s"

    degrade = "0"
    if mode == "s":
        degrade = input("Choose distortions [0,1,2,3,12,23,123] → ").strip()
        if degrade == "":
            degrade = "0"

    gt, degraded, meta = create_test_sample(mode, degrade)

    print("\n--- META DATA ---")
    for k, v in meta.items():
        print(f"{k}: {v}")

    # Display
    if gt is None:
        plt.imshow(degraded)
        plt.title("Natural degraded (no GT)")
        plt.axis("off")
    else:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(gt)
        plt.title("GT Clean")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(degraded)
        plt.title(f"Degraded {meta['code']}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
