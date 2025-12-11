import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models
from numpy.fft import fft2, ifft2

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None
    print("WARNING: skimage not installed → SSIM will be None.")

try:
    import lpips
except ImportError:
    lpips = None
    print("WARNING: lpips not installed → LPIPS will be None.")

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("WARNING: ultralytics not installed → YOLO metrics will be None.")


# ============================================================
# CONFIG
# ============================================================
BASE_DIR = r"C:\Users\khump\OneDrive\Desktop\ECE253Project"  # update as needed

MULTI_DIR   = os.path.join(BASE_DIR, "multi_degraded_100k")
LABELS_CSV  = os.path.join(MULTI_DIR, "labels_multitask.csv")

CLEAN_IMAGE_PATH   = None
NATURAL_IMAGE_PATH = None

MODEL_PATH   = os.path.join(BASE_DIR, "multitask_resnet18_blur_fog_lowlight.pth")
YOLO_WEIGHTS = os.path.join(BASE_DIR, "yolov8n.pt")

IMG_SIZE = 224

# Normalization ranges for regression targets
L_MIN, L_MAX         = 0.0, 60.0
THETA_MIN, THETA_MAX = 0.0, 180.0
THETA_RANGE          = THETA_MAX - THETA_MIN

# Blur module thresholds (SMART pipeline)
BLUR_PROB_THRESH = 0.6
L_MIN_DEBLUR     = 30.0

# Wiener constant used in pipeline
K_WEINER_DEFAULT = 0.01

# Fog module threshold (SMART pipeline)
FOG_PROB_THRESH = 0.6

# Low-light module threshold (SMART pipeline)
LOWLIGHT_PROB_THRESH = 0.6

# How many random test images to evaluate
N_EVAL_SAMPLES = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# MULTITASK MODEL
# ============================================================
class MultiTaskResNet(nn.Module):
    """
    Same as in training:
      - cls_head: 3 logits (blur, fog, lowlight)
      - reg_head: 2 outputs (L_norm, theta_norm)
    """
    def __init__(self, backbone, num_feats):
        super().__init__()
        self.backbone = backbone
        self.backbone.fc = nn.Identity()

        # classification: [B, num_feats] -> [B, 3]
        self.cls_head = nn.Linear(num_feats, 3)

        # regression: [B, num_feats] -> [B, 2]
        self.reg_head = nn.Sequential(
            nn.Linear(num_feats, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # [L_norm, theta_norm]
        )

    def forward(self, x):
        feats = self.backbone(x)
        cls_logits = self.cls_head(feats)  # [B,3]
        reg_out    = self.reg_head(feats)  # [B,2]
        return cls_logits, reg_out


print("Loading multitask model from:", MODEL_PATH)
_backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
_num_feats = _backbone.fc.in_features
model = MultiTaskResNet(_backbone, _num_feats).to(device)

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded.\n")


# ============================================================
# METRIC MODELS (LPIPS, YOLO)
# ============================================================
lpips_model = None
if lpips is not None:
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()

yolo_model = None
if YOLO is not None and os.path.exists(YOLO_WEIGHTS):
    yolo_model = YOLO(YOLO_WEIGHTS)
    print("Loaded YOLO model from:", YOLO_WEIGHTS)
elif YOLO is not None:
    print("WARNING: YOLO weights not found at", YOLO_WEIGHTS, "→ YOLO metrics disabled.")


# ============================================================
# TRANSFORMS
# ============================================================
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ============================================================
# UTILS: DENORM L, THETA
# ============================================================
def denorm(L_norm, theta_norm):
    """
    Convert normalized L, theta back to:
      L in [L_MIN, L_MAX]
      theta in [THETA_MIN, THETA_MAX]
    """
    L = L_norm * (L_MAX - L_MIN) + L_MIN
    theta = theta_norm * (THETA_MAX - THETA_MIN) + THETA_MIN
    return L, theta


# ============================================================
# SYNTHETIC DEGRADATIONS (for DEMO image)
# ============================================================
def motion_blur_degrade(img_rgb, L=15, theta=45):
    ksize = max(3, int(L))
    if ksize % 2 == 0:
        ksize += 1

    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0

    M = cv2.getRotationMatrix2D((ksize/2 - 0.5, ksize/2 - 0.5), theta, 1.0)
    kernel = cv2.warpAffine(kernel, M, (ksize, ksize),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)

    s = kernel.sum()
    if s > 0:
        kernel /= s

    blurred = cv2.filter2D(img_rgb, -1, kernel, borderType=cv2.BORDER_REFLECT)
    return blurred


def add_fog_degrade(img, strength=0.5):
    img = img.astype(np.float32) / 255.0
    A = np.array([0.9, 0.9, 0.9], dtype=np.float32)
    fogged = (1 - strength) * img + strength * A
    return np.clip(fogged * 255, 0, 255).astype(np.uint8)


def add_low_light_degrade(img, brightness=0.4, gamma=1.5):
    img = img.astype(np.float32) / 255.0
    img = (img * brightness) ** gamma
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def create_test_sample(mode, degrade_code):
    """
    Demo image generator.

    mode:
      's' → synthetic (start from CLEAN_IMAGE_PATH)
      'n' → natural  (start from NATURAL_IMAGE_PATH, GT=None)
    degrade_code ('0','1','2','3','12','23','123'):
      1 → blur
      2 → fog
      3 → low-light

    Returns:
      gt_rgb (or None), degraded_rgb, meta dict
    """
    meta = {
        "mode": mode,
        "code": degrade_code,
        "blur": False,
        "fog": False,
        "lowlight": False,
        "blur_L": None,
        "blur_theta": None,
        "fog_strength": None,
        "brightness": None,
        "gamma": None,
    }

    if mode == "n":
        img_bgr = cv2.imread(NATURAL_IMAGE_PATH)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read NATURAL_IMAGE_PATH: {NATURAL_IMAGE_PATH}")
        degraded = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt_rgb = None
        return gt_rgb, degraded, meta

    # synthetic: start from CLEAN_IMAGE_PATH (chosen from test set)
    img_bgr = cv2.imread(CLEAN_IMAGE_PATH)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read CLEAN_IMAGE_PATH: {CLEAN_IMAGE_PATH}")
    gt_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    degraded = gt_rgb.copy()

    if degrade_code != "0":
        if "1" in degrade_code:
            L = float(np.random.uniform(5, 60))
            T = float(np.random.uniform(0, 180))
            degraded = motion_blur_degrade(degraded, L=L, theta=T)
            meta["blur"] = True
            meta["blur_L"] = L
            meta["blur_theta"] = T

        if "2" in degrade_code:
            S = float(np.random.uniform(0.3, 0.8))
            degraded = add_fog_degrade(degraded, strength=S)
            meta["fog"] = True
            meta["fog_strength"] = S

        if "3" in degrade_code:
            B = float(np.random.uniform(0.2, 0.5))
            G = float(np.random.uniform(1.2, 1.8))
            degraded = add_low_light_degrade(degraded, brightness=B, gamma=G)
            meta["lowlight"] = True
            meta["brightness"] = B
            meta["gamma"] = G

    return gt_rgb, degraded, meta


# ============================================================
# BLUR MODULE UTILITIES (PSF / DECONV)
# ============================================================
def motion_psf(L, theta_deg, shape_hw):
    H, W = shape_hw
    ksize = int(max(3, L))
    if ksize % 2 == 0:
        ksize += 1

    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0

    M = cv2.getRotationMatrix2D((ksize/2 - 0.5, ksize/2 - 0.5), theta_deg, 1.0)
    kernel = cv2.warpAffine(
        kernel, M, (ksize, ksize),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    kernel = cv2.GaussianBlur(kernel, (0, 0), sigmaX=ksize/6.0)

    s = kernel.sum()
    if s > 0:
        kernel /= s

    P = np.zeros((H, W), dtype=np.float32)
    ph, pw = kernel.shape
    P[:ph, :pw] = kernel
    P = np.roll(P, -ph // 2, axis=0)
    P = np.roll(P, -pw // 2, axis=1)
    return P


def regular_deconvolve_y(img_rgb_uint8, L, theta, eps=1e-3):
    img_bgr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    Yf = Y.astype(np.float32) / 255.0
    H, W = Yf.shape
    P = motion_psf(L, theta, (H, W))

    G = fft2(Yf)
    Hf = fft2(P)

    F_est = G / (Hf + eps)
    y_rec = np.real(ifft2(F_est))
    y_rec = np.clip(y_rec, 0.0, 1.0)
    Y_rec = (y_rec * 255.0).astype(np.uint8)

    out_ycrcb = cv2.merge((Y_rec, Cr, Cb))
    out_bgr = cv2.cvtColor(out_ycrcb, cv2.COLOR_YCrCb2BGR)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    return out_rgb


def wiener_deconvolve_y(img_rgb_uint8, L, theta, K=1e-2):
    img_bgr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    Yf = Y.astype(np.float32) / 255.0
    H, W = Yf.shape
    P = motion_psf(L, theta, (H, W))

    G = fft2(Yf)
    Hf = fft2(P)

    H_conj = np.conj(Hf)
    denom = np.abs(Hf)**2 + K
    F_est = (H_conj / denom) * G

    y_rec = np.real(ifft2(F_est))
    y_rec = np.clip(y_rec, 0.0, 1.0)
    Y_rec = (y_rec * 255.0).astype(np.uint8)

    out_ycrcb = cv2.merge((Y_rec, Cr, Cb))
    out_bgr = cv2.cvtColor(out_ycrcb, cv2.COLOR_YCrCb2BGR)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    return out_rgb


# ============================================================
# FOG MODULE UTILITIES (HISTEQ)
# ============================================================
def equalize_histogram_rgb(image):
    """
    Apply histogram equalization channel-wise in RGB.
    """
    if image.ndim == 2:
        return cv2.equalizeHist(image)

    if image.ndim == 3 and image.shape[2] == 3:
        equalized = np.zeros_like(image)
        for i in range(3):
            equalized[:, :, i] = cv2.equalizeHist(image[:, :, i])
        return equalized

    raise ValueError("Input must be grayscale or 3-channel RGB.")


# ============================================================
# LOW-LIGHT MODULE UTILITIES (Retinex-like)
# ============================================================
def retinex_dual_illumination(img_rgb,
                              sigma1=15,
                              sigma2=80,
                              clip_range=0.3,
                              blend=0.5,
                              gamma=1.0):
    img = img_rgb.astype(np.float32) / 255.0
    eps = 1e-3

    I1 = cv2.GaussianBlur(img, (0, 0), sigma1)
    I2 = cv2.GaussianBlur(img, (0, 0), sigma2)

    R1 = np.log(img + eps) - np.log(I1 + eps)
    R2 = np.log(img + eps) - np.log(I2 + eps)
    R  = 0.5 * (R1 + R2)

    R = np.clip(R, -clip_range, clip_range)

    R_min = R.min(axis=(0, 1), keepdims=True)
    R_max = R.max(axis=(0, 1), keepdims=True)
    Rn = (R - R_min) / (R_max - R_min + eps)

    Rn = np.power(Rn, gamma)

    enhanced = (1.0 - blend) * img + blend * Rn
    enhanced = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
    return enhanced


# ============================================================
# SMALL METRIC HELPERS
# ============================================================
def to_gray(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


def tenengrad_sharpness(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag2 = gx**2 + gy**2
    return float(mag2.mean())


def brightness_mean(gray):
    return float(gray.mean())


def psnr(img1, img2):
    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)
    mse = np.mean((img1_f - img2_f) ** 2)
    if mse == 0:
        return 99.0
    return 20 * np.log10(255.0 / np.sqrt(mse))


def ssim_metric(img1, img2):
    if ssim is None:
        return None
    g1 = to_gray(img1)
    g2 = to_gray(img2)
    return float(ssim(g1, g2, data_range=255))


def lpips_metric(img1, img2):
    if lpips_model is None:
        return None
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)

    def prep(im):
        im = im.astype(np.float32) / 255.0
        im = 2.0 * im - 1.0
        im = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).to(device)
        return im

    t1 = prep(img1)
    t2 = prep(img2)
    with torch.no_grad():
        d = lpips_model(t1, t2)
    return float(d.item())


def yolo_stats(img_rgb):
    if yolo_model is None:
        return None, None

    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    results = yolo_model(bgr, verbose=False)
    r = results[0]
    boxes = getattr(r, "boxes", None)
    if boxes is None or boxes.xyxy is None or boxes.xyxy.shape[0] == 0:
        return 0, 0.0
    conf = boxes.conf.cpu().numpy()
    return int(conf.size), float(conf.mean())


# ============================================================
# MODEL PREDICTION HELPERS
# ============================================================
def predict_all(img_rgb):
    """
    Return blur/fog/lowlight probabilities + denormalized L, theta.
    """
    pil_img = Image.fromarray(img_rgb)
    x = eval_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        cls_logits, reg_out = model(x)
        probs = torch.sigmoid(cls_logits)[0]  # [3]
        L_norm     = reg_out[0, 0]
        theta_norm = reg_out[0, 1]

    blur_prob = float(probs[0].item())
    fog_prob  = float(probs[1].item())
    low_prob  = float(probs[2].item())

    L_pred, theta_pred = denorm(L_norm, theta_norm)
    L_pred     = float(L_pred.item())
    theta_pred = float(theta_pred.item())
    theta_pred = max(0.0, min(180.0, theta_pred))

    return blur_prob, fog_prob, low_prob, L_pred, theta_pred


def predict_L_theta_only(img_rgb):
    """
    Use the multitask model only to get L, theta (regression outputs).
    Ignore classification logits (for naive pipeline).
    """
    pil_img = Image.fromarray(img_rgb)
    x = eval_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        cls_logits, reg_out = model(x)  # cls_logits is ignored
        L_norm     = reg_out[0, 0]
        theta_norm = reg_out[0, 1]

    L_pred, theta_pred = denorm(L_norm, theta_norm)
    L_pred     = float(L_pred.item())
    theta_pred = float(theta_pred.item())
    theta_pred = max(0.0, min(180.0, theta_pred))

    return L_pred, theta_pred


# ============================================================
# BLUR MODULE (SMART)
# ============================================================
def run_blur_module(input_rgb, gt_rgb=None, show_plots=True):
    blur_prob, fog_prob, low_prob, L_pred, theta_pred = predict_all(input_rgb)

    print("\n[BlurModule] Predictions:")
    print(f"  blur_prob  = {blur_prob:.3f}")
    print(f"  fog_prob   = {fog_prob:.3f}")
    print(f"  low_prob   = {low_prob:.3f}")
    print(f"  L_pred     = {L_pred:.2f} px")
    print(f"  theta_pred = {theta_pred:.2f} °")

    if (blur_prob < BLUR_PROB_THRESH) or (L_pred < L_MIN_DEBLUR):
        print("[BlurModule] No strong motion blur detected → pass-through.")
        meta = {
            "blur_module_applied": False,
            "blur_prob": blur_prob,
            "fog_prob": fog_prob,
            "low_prob": low_prob,
            "L_pred": L_pred,
            "theta_pred": theta_pred,
            "K_used": None,
        }
        debug = {"regular": None, "wiener": None}
        out_rgb = input_rgb.copy()

        if show_plots:
            plt.figure(figsize=(6, 6))
            plt.imshow(out_rgb)
            plt.title("BlurModule: no deblurring (pass-through)")
            plt.axis("off")
            plt.show()

        return out_rgb, gt_rgb, meta, debug

    print(f"[BlurModule] Motion blur detected (L_pred={L_pred:.2f}) → deblurring.")

    reg_rgb = regular_deconvolve_y(input_rgb, L_pred, theta_pred, eps=1e-3)
    wiener_default = wiener_deconvolve_y(input_rgb, L_pred, theta_pred, K=K_WEINER_DEFAULT)

    meta = {
        "blur_module_applied": True,
        "blur_prob": blur_prob,
        "fog_prob": fog_prob,
        "low_prob": low_prob,
        "L_pred": L_pred,
        "theta_pred": theta_pred,
        "K_used": K_WEINER_DEFAULT,
    }

    debug = {
        "regular": reg_rgb,
        "wiener": wiener_default,
    }

    if show_plots:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1); plt.imshow(input_rgb);      plt.title("Input (possibly blurred)"); plt.axis("off")
        plt.subplot(1, 3, 2); plt.imshow(reg_rgb);        plt.title("Regular Deconv");           plt.axis("off")
        plt.subplot(1, 3, 3); plt.imshow(wiener_default); plt.title(f"Wiener K={K_WEINER_DEFAULT:.3f} (chosen)"); plt.axis("off")
        plt.tight_layout()
        plt.show()

    return wiener_default, gt_rgb, meta, debug


# ============================================================
# FOG MODULE (SMART, HISTEQ)
# ============================================================
def run_fog_module(input_rgb, gt_rgb=None, show_plots=True):
    blur_prob, fog_prob, low_prob, _, _ = predict_all(input_rgb)

    print("\n[FogModule] Predictions:")
    print(f"  blur_prob = {blur_prob:.3f}")
    print(f"  fog_prob  = {fog_prob:.3f}")
    print(f"  low_prob  = {low_prob:.3f}")

    if fog_prob < FOG_PROB_THRESH:
        print("[FogModule] No strong fog detected → pass-through.")
        meta = {
            "fog_module_applied": False,
            "fog_prob": fog_prob,
            "blur_prob": blur_prob,
            "low_prob": low_prob,
            "method": "none",
        }
        debug = {"histeq": None}

        if show_plots:
            plt.figure(figsize=(6, 6))
            plt.imshow(input_rgb)
            plt.title("FogModule: no defogging (pass-through)")
            plt.axis("off")
            plt.show()

        return input_rgb.copy(), gt_rgb, meta, debug

    print(f"[FogModule] Fog detected (fog_prob={fog_prob:.3f}) → histogram equalization.")

    hist_img = equalize_histogram_rgb(input_rgb)

    gray_in    = to_gray(input_rgb)
    gray_hist  = to_gray(hist_img)

    ten_in    = tenengrad_sharpness(gray_in)
    ten_hist  = tenengrad_sharpness(gray_hist)

    br_in    = brightness_mean(gray_in)
    br_hist  = brightness_mean(gray_hist)

    print("\n[FogModule] Tenengrad / Brightness (for debugging/logging):")
    print(f"  Input : Ten={ten_in:.1f},  Bright={br_in:.1f}")
    print(f"  Hist : Ten={ten_hist:.1f}, Bright={br_hist:.1f}")

    out_rgb = hist_img

    meta = {
        "fog_module_applied": True,
        "fog_prob": fog_prob,
        "blur_prob": blur_prob,
        "low_prob": low_prob,
        "method": "histeq_rgb",
        "ten_input": ten_in,
        "ten_histeq": ten_hist,
        "bright_input": br_in,
        "bright_histeq": br_hist,
    }

    debug = {
        "histeq": hist_img,
    }

    if show_plots:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1); plt.imshow(input_rgb);  plt.title("Input (possibly foggy)"); plt.axis("off")
        plt.subplot(1, 2, 2); plt.imshow(hist_img);   plt.title("Histogram Equalization (chosen)"); plt.axis("off")
        plt.tight_layout()
        plt.show()

    return out_rgb, gt_rgb, meta, debug


# ============================================================
# LOW-LIGHT MODULE (SMART)
# ============================================================
def run_lowlight_module(input_rgb, gt_rgb=None, show_plots=True):
    blur_prob, fog_prob, low_prob, _, _ = predict_all(input_rgb)

    print("\n[LowLightModule] Predictions:")
    print(f"  blur_prob = {blur_prob:.3f}")
    print(f"  fog_prob  = {fog_prob:.3f}")
    print(f"  low_prob  = {low_prob:.3f}")

    gray = to_gray(input_rgb)
    mean_brightness = brightness_mean(gray)
    print(f"  mean brightness = {mean_brightness:.1f}")

    if (low_prob < LOWLIGHT_PROB_THRESH) or (mean_brightness > 90):
        print("[LowLightModule] No strong low-light (or already bright) → pass-through.")
        meta = {
            "lowlight_module_applied": False,
            "low_prob": low_prob,
            "blur_prob": blur_prob,
            "fog_prob": fog_prob,
            "method": "none",
            "mean_brightness": mean_brightness,
        }
        debug = {"retinex": None}

        if show_plots:
            plt.figure(figsize=(6, 6))
            plt.imshow(input_rgb)
            plt.title("LowLightModule: no enhancement (pass-through)")
            plt.axis("off")
            plt.show()

        return input_rgb.copy(), gt_rgb, meta, debug

    print(f"[LowLightModule] Low-light detected (low_prob={low_prob:.3f}) → Retinex enhancement.")

    ret_img = retinex_dual_illumination(
        input_rgb,
        sigma1=15,
        sigma2=80,
        clip_range=0.25,
        blend=0.4,
        gamma=1.0,
    )

    meta = {
        "lowlight_module_applied": True,
        "low_prob": low_prob,
        "blur_prob": blur_prob,
        "fog_prob": fog_prob,
        "method": "dual_illumination_retinex",
        "mean_brightness_before": mean_brightness,
        "mean_brightness_after": brightness_mean(to_gray(ret_img)),
    }

    debug = {"retinex": ret_img}

    if show_plots:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1); plt.imshow(input_rgb); plt.title("Input (possibly low-light)"); plt.axis("off")
        plt.subplot(1, 2, 2); plt.imshow(ret_img);   plt.title("Retinex Enhanced (blended)"); plt.axis("off")
        plt.tight_layout()
        plt.show()

    return ret_img, gt_rgb, meta, debug


# ============================================================
# NAIVE PIPELINE (NO GATING, ALWAYS APPLY MODULES)
# ============================================================
def run_naive_full_pipeline(input_rgb, show_plots=False):
    """
    Naive baseline:
      - Always apply blur deconvolution using the network's predicted L, theta,
        with NO thresholds or gating.
      - Fog stage: always apply histogram equalization (no prediction at all).
      - Low-light stage: always apply Retinex-based enhancement (no prediction).
    """
    L_pred, theta_pred = predict_L_theta_only(input_rgb)

    print("\n[NaivePipeline]")
    print("  Using predicted motion-blur parameters without thresholds:")
    print(f"  L_pred     = {L_pred:.2f} px")
    print(f"  theta_pred = {theta_pred:.2f} °")

    # 1) ALWAYS deblur with Wiener filter
    blur_out = wiener_deconvolve_y(input_rgb, L_pred, theta_pred, K=K_WEINER_DEFAULT)

    # 2) ALWAYS apply histogram equalization
    fog_out = equalize_histogram_rgb(blur_out)

    # 3) ALWAYS apply Retinex enhancement
    low_out = retinex_dual_illumination(
        fog_out,
        sigma1=15,
        sigma2=80,
        clip_range=0.25,
        blend=0.4,
        gamma=1.0,
    )

    if show_plots:
        plt.figure(figsize=(18, 4))
        plt.subplot(1, 4, 1); plt.imshow(input_rgb); plt.title("Naive: input");     plt.axis("off")
        plt.subplot(1, 4, 2); plt.imshow(blur_out);  plt.title("Naive: deblur");    plt.axis("off")
        plt.subplot(1, 4, 3); plt.imshow(fog_out);   plt.title("Naive: histeq");    plt.axis("off")
        plt.subplot(1, 4, 4); plt.imshow(low_out);   plt.title("Naive: Retinex");   plt.axis("off")
        plt.tight_layout()
        plt.show()

    return low_out


# ============================================================
# PER-IMAGE METRICS (NO PLOTS)
# ============================================================
def compute_metrics_for_variants(gt_rgb, variants_dict):
    """
    variants_dict: name -> image (RGB uint8)
    Metrics that use GT (clean): PSNR, SSIM, LPIPS.
    Others are per-image: Tenengrad, brightness, YOLO stats.
    """
    metrics = {}

    for name, img in variants_dict.items():
        m = {}

        if gt_rgb is not None:
            psnr_val  = psnr(gt_rgb, img)
            ssim_val  = ssim_metric(gt_rgb, img)
            lpips_val = lpips_metric(gt_rgb, img)
        else:
            psnr_val = ssim_val = lpips_val = None

        m["psnr"]   = psnr_val
        m["ssim"]   = ssim_val
        m["lpips"]  = lpips_val

        gray = to_gray(img)
        m["tenengrad"] = tenengrad_sharpness(gray)
        m["brightness"] = brightness_mean(gray)

        if yolo_model is not None:
            cnt, conf = yolo_stats(img)
            m["yolo_count"] = cnt
            m["yolo_conf"]  = conf
        else:
            m["yolo_count"] = None
            m["yolo_conf"]  = None

        metrics[name] = m

    return metrics


# ============================================================
# AGGREGATE METRICS PLOTTING
# (Only PSNR/SSIM/LPIPS + YOLO metrics; no sharpness/brightness figure)
# ============================================================
def plot_aggregate_metrics(agg_metrics):
    """
    agg_metrics: dict[variant][metric_name] -> list of values
      e.g. agg_metrics["smart_full"]["psnr"] = [ ..., ... ]

    This function:
      - computes mean ± std per (variant, metric)
      - creates figures:
          1) reference-based metrics: PSNR, SSIM, LPIPS
          2) YOLO metrics: detection count, mean confidence
    """
    variants = list(agg_metrics.keys())
    metric_names = list(next(iter(agg_metrics.values())).keys())

    base_colors = [
        "#4C72B0",  # blue
        "#55A868",  # green
        "#C44E52",  # red
        "#8172B3",  # purple
        "#CCB974",  # yellow/brown
        "#64B5CD",  # teal
        "#8C564B",  # brown
    ]

    def metric_stats(metric):
        """Return (means, stds) across variants for a given metric."""
        means, stds = [], []
        for v in variants:
            vals = np.array([x for x in agg_metrics[v][metric] if x is not None])
            if vals.size == 0:
                means.append(np.nan)
                stds.append(0.0)
            else:
                means.append(float(vals.mean()))
                stds.append(float(vals.std()))
        return means, stds

    def plot_metric_group(title, metrics_in_group, pretty_names, ylabel=None):
        # Filter out metrics that have all NaNs
        filtered = []
        for m, nice in zip(metrics_in_group, pretty_names):
            if m in metric_names:
                means, _ = metric_stats(m)
                if not all(np.isnan(means)):
                    filtered.append((m, nice))

        if len(filtered) == 0:
            return

        n_metrics = len(filtered)
        x = np.arange(len(variants))
        width = 0.15 if n_metrics > 3 else 0.2

        plt.figure(figsize=(6 + 2 * n_metrics, 5))
        plt.title(title, fontsize=14, fontweight="bold")

        for i, (mname, pretty) in enumerate(filtered):
            means, stds = metric_stats(mname)
            offset = (i - (n_metrics - 1) / 2) * width
            bar_positions = x + offset

            color = base_colors[i % len(base_colors)]

            plt.bar(
                bar_positions,
                means,
                width,
                yerr=stds,
                capsize=4,
                label=pretty,
                color=color,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.7,
            )

            # numeric labels
            for bx, by in zip(bar_positions, means):
                if np.isnan(by):
                    continue
                if mname in ["psnr", "brightness", "tenengrad", "yolo_count"]:
                    txt = f"{by:.2f}"
                else:
                    txt = f"{by:.3f}"
                plt.text(
                    bx,
                    by,
                    txt,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.xticks(x, variants, rotation=20, ha="right", fontsize=10)
        if ylabel is not None:
            plt.ylabel(ylabel, fontsize=12)

        plt.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.7)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()

    # 1) Reference-based metrics
    plot_metric_group(
        title="Reference-based Metrics (averaged over test images)",
        metrics_in_group=["psnr", "ssim", "lpips"],
        pretty_names=["PSNR (dB)", "SSIM", "LPIPS (lower = better)"],
        ylabel="Score",
    )

    # 2) YOLO metrics
    plot_metric_group(
        title="YOLO Detection Metrics",
        metrics_in_group=["yolo_count", "yolo_conf"],
        pretty_names=["YOLO Detection Count", "YOLO Mean Confidence"],
        ylabel="Value",
    )


# ============================================================
# HELPER: GET GT PATH FROM DEG PATH
# ============================================================
def deduce_gt_path_from_deg_path(deg_path):
    """
    deg_path: e.g. .../test/DEG1_23_xxxxx.jpg
    GT path should be .../test/CLEAN_xxxxx.jpg
    """
    folder = os.path.dirname(deg_path)
    fname  = os.path.basename(deg_path)

    if fname.startswith("CLEAN_"):
        return deg_path

    parts = fname.split("_", 2)
    if len(parts) < 3:
        return deg_path  # fallback: no clean twin

    orig_name = parts[2]
    clean_name = "CLEAN_" + orig_name
    return os.path.join(folder, clean_name)


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    global CLEAN_IMAGE_PATH, NATURAL_IMAGE_PATH

    print("\n=== Unified Pipeline: Demo + Metrics (Clean vs Degraded vs Smart vs Naive) ===")
    print("Reading labels CSV from:", LABELS_CSV)

    df = pd.read_csv(LABELS_CSV)
    df_test = df[df["set"] == "test"].copy()
    print(f"Test set rows in CSV: {len(df_test)}")

    if len(df_test) == 0:
        raise RuntimeError("No test rows found in labels_multitask.csv (set == 'test').")

    # Pick a clean-ish test image for the DEMO
    clean_candidates = [
        path for path in df_test["filepath"]
        if os.path.basename(path).startswith("CLEAN_")
    ]

    if len(clean_candidates) > 0:
        CLEAN_IMAGE_PATH = np.random.choice(clean_candidates)
    else:
        CLEAN_IMAGE_PATH = df_test.sample(1, random_state=42)["filepath"].iloc[0]

    NATURAL_IMAGE_PATH = CLEAN_IMAGE_PATH  # can customize if you want a different natural image

    print("\n[Demo] Using this test image as base for degradation:")
    print("  CLEAN_IMAGE_PATH   =", CLEAN_IMAGE_PATH)
    print("  NATURAL_IMAGE_PATH =", NATURAL_IMAGE_PATH)

    # DEMO: user choice
    mode = input("Use synthetic (s) or natural (n) image for DEMO? [s/n]: ").strip().lower()
    if mode not in ["s", "n"]:
        print("Invalid mode. Use 's' or 'n'.")
        return

    degrade_code = "0"
    if mode == "s":
        degrade_code = input("Degradation code for DEMO [0,1,2,3,12,23,123]: ").strip()

    # 1) Demo generation
    gt_demo, degraded_demo, meta_gen_demo = create_test_sample(mode, degrade_code)
    print("\n[Demo Generator] Meta:")
    print(json.dumps(meta_gen_demo, indent=2))

    # 2) SMART full pipeline demo
    blur_demo, gt_blur_demo, _, _ = run_blur_module(
        degraded_demo, gt_demo, show_plots=True
    )
    fog_demo, gt_fog_demo, _, _ = run_fog_module(
        blur_demo, gt_blur_demo, show_plots=True
    )
    smart_demo, gt_smart_demo, _, _ = run_lowlight_module(
        fog_demo, gt_fog_demo, show_plots=True
    )

    # 3) NAIVE full pipeline demo
    naive_demo = run_naive_full_pipeline(degraded_demo, show_plots=True)

    # 4) Summary plots for the demo
    plt.figure(figsize=(22, 4))

    cols = 4 if gt_demo is None else 5
    idx = 1

    if gt_demo is not None:
        plt.subplot(1, cols, idx); idx += 1
        plt.imshow(gt_demo)
        plt.title("GT (clean) - DEMO")
        plt.axis("off")

    plt.subplot(1, cols, idx); idx += 1
    plt.imshow(degraded_demo)
    plt.title(f"Degraded (code={meta_gen_demo['code']})")
    plt.axis("off")

    plt.subplot(1, cols, idx); idx += 1
    plt.imshow(smart_demo)
    plt.title("Smart Pipeline Output")
    plt.axis("off")

    plt.subplot(1, cols, idx); idx += 1
    plt.imshow(naive_demo)
    plt.title("Naive Pipeline Output")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # ---------- Ask whether to run full evaluation ----------
    run_eval = input("\nAlso run evaluation on random degraded test images? [y/n]: ").strip().lower()
    if run_eval != "y":
        print("Skipping test-set evaluation. Demo finished.")
        return

    # ---------- Evaluation on REAL degraded test images ----------
    print("\n=== Evaluation on degraded test images from multi_degraded_100k/test ===")

    # Use test rows that do NOT have CLEAN_ prefix as degraded
    mask_clean = df_test["filepath"].str.contains("CLEAN_")
    df_deg = df_test[~mask_clean].copy()

    if len(df_deg) == 0:
        print("No non-CLEAN_ test rows; using all test rows for evaluation.")
        df_deg = df_test.copy()
    else:
        print(f"Using {len(df_deg)} degraded test rows (non-CLEAN_).")

    n_samples = min(N_EVAL_SAMPLES, len(df_deg))
    print(f"Evaluating on {n_samples} random degraded test images...")

    sampled_rows = df_deg.sample(n_samples, random_state=123)

    # We only care about these four variants:
    # 1) clean (GT)
    # 2) degraded
    # 3) smart_full (smart pipeline output)
    # 4) naive_full (naive pipeline output)
    variants = ["clean", "degraded", "smart_full", "naive_full"]
    metrics_names = ["psnr", "ssim", "lpips",
                     "tenengrad", "brightness",
                     "yolo_count", "yolo_conf"]

    agg_metrics = {
        v: {m: [] for m in metrics_names} for v in variants
    }

    for idx_row, row in sampled_rows.iterrows():
        deg_path = row["filepath"]
        gt_path  = deduce_gt_path_from_deg_path(deg_path)

        deg_bgr = cv2.imread(deg_path)
        if deg_bgr is None:
            print(f"[Eval] WARN: could not read degraded image: {deg_path}")
            continue
        deg_rgb = cv2.cvtColor(deg_bgr, cv2.COLOR_BGR2RGB)

        gt_bgr = cv2.imread(gt_path)
        if gt_bgr is None:
            print(f"[Eval] WARN: could not read GT image: {gt_path}, using degraded as GT fallback.")
            gt_rgb = deg_rgb.copy()
        else:
            gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)

        # SMART pipeline WITHOUT plots
        blur_out, gt1, _, _ = run_blur_module(deg_rgb, gt_rgb, show_plots=False)
        fog_out,  gt2, _, _ = run_fog_module(blur_out, gt1, show_plots=False)
        smart_out, gt3, _, _ = run_lowlight_module(fog_out, gt2, show_plots=False)

        # NAIVE pipeline WITHOUT plots
        naive_out = run_naive_full_pipeline(deg_rgb, show_plots=False)

        variants_imgs = {
            "clean":      gt3,       # use final GT reference
            "degraded":   deg_rgb,
            "smart_full": smart_out,
            "naive_full": naive_out,
        }

        img_metrics = compute_metrics_for_variants(gt3, variants_imgs)

        for vname in variants:
            mvals = img_metrics[vname]
            for mname in metrics_names:
                agg_metrics[vname][mname].append(mvals.get(mname, None))

    # Print aggregated means
    print("\n=== AGGREGATED METRICS OVER TEST IMAGES ===")
    for v in variants:
        print(f"\nVariant: {v}")
        for mname in metrics_names:
            vals = [x for x in agg_metrics[v][mname] if x is not None]
            if len(vals) == 0:
                mean_str = "N/A"
            else:
                mean_val = float(np.mean(vals))
                if mname in ["psnr", "brightness", "tenengrad"]:
                    mean_str = f"{mean_val:.2f}"
                elif mname in ["ssim", "lpips", "yolo_conf"]:
                    mean_str = f"{mean_val:.4f}"
                else:
                    mean_str = f"{mean_val:.3f}"
            print(f"  {mname:12s}: {mean_str}")

    # Plot bar charts of averages (no separate sharpness/brightness figure)
    plot_aggregate_metrics(agg_metrics)


if __name__ == "__main__":
    main()
