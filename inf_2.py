import os
import random
import glob

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models
from numpy.fft import fft2, ifft2

# Optional: SSIM (skimage)
try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False

# Optional: LPIPS
try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

# YOLO (Ultralytics)
from ultralytics import YOLO

# ================== CONFIG ==================
BASE_DIR       = r"C:\ECE253Project"
DATA_DIR       = os.path.join(BASE_DIR, "blurred_50_50")
LABELS_CSV     = os.path.join(DATA_DIR, "labels.csv")
MODEL_PATH     = os.path.join(BASE_DIR, "blur_multitask_resnet18_circ_stepLR.pth")

# Original clean 100k dataset root
ORIG_100K_DIR  = os.path.join(BASE_DIR, "100k")

IMG_SIZE       = 224
L_MIN, L_MAX   = 0.0, 60.0
THETA_MIN, THETA_MAX = 0.0, 180.0

# Try multiple Wiener regularization constants
K_LIST = [1e-4, 5e-3, 1e-2, 3e-2, 1e-1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ================== MODEL DEFINITION (MATCH TRAIN) ==================
class MultiTaskResNet(nn.Module):
    def __init__(self, backbone, num_feats):
        super().__init__()
        self.backbone = backbone
        self.backbone.fc = nn.Identity()

        # blur classifier head (logit)
        self.blur_head = nn.Linear(num_feats, 1)

        # regression head for L, theta
        self.reg_head = nn.Sequential(
            nn.Linear(num_feats, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # [L_norm, theta_norm]
        )

    def forward(self, x):
        feats = self.backbone(x)
        blur_logit = self.blur_head(feats).squeeze(1)
        reg_out = self.reg_head(feats)
        return blur_logit, reg_out


def denorm(L_norm, theta_norm):
    L = L_norm * (L_MAX - L_MIN) + L_MIN
    theta = theta_norm * (THETA_MAX - THETA_MIN) + THETA_MIN
    return L, theta


print("Loading motion-blur CNN from:", MODEL_PATH)
backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_feats = backbone.fc.in_features
model = MultiTaskResNet(backbone, num_feats).to(device)

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print("Motion-blur model loaded.\n")


# ================== TRANSFORM (MATCH TRAIN) ==================
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])


# ================== PSF / DECONV UTILITIES ==================
def motion_psf(L, theta_deg, shape_hw):
    """
    Create a smoothed motion PSF, padded to image shape and centered for FFT.
    """
    H, W = shape_hw
    ksize = int(max(3, L))
    if ksize % 2 == 0:
        ksize += 1

    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0

    M = cv2.getRotationMatrix2D((ksize / 2 - 0.5, ksize / 2 - 0.5), theta_deg, 1.0)
    kernel = cv2.warpAffine(
        kernel, M, (ksize, ksize),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    # Slight smoothing to reduce ringing
    kernel = cv2.GaussianBlur(kernel, (0, 0), sigmaX=ksize / 6.0)

    s = kernel.sum()
    if s > 0:
        kernel /= s

    P = np.zeros((H, W), dtype=np.float32)
    ph, pw = kernel.shape
    P[:ph, :pw] = kernel
    P = np.roll(P, -ph // 2, axis=0)
    P = np.roll(P, -pw // 2, axis=1)
    return P


def regular_deconvolve_y_channel(img_rgb_uint8, L, theta, eps=1e-3):
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


def wiener_deconvolve_y_channel(img_rgb_uint8, L, theta, K=1e-2):
    img_bgr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    Yf = Y.astype(np.float32) / 255.0
    H, W = Yf.shape
    P = motion_psf(L, theta, (H, W))

    G = fft2(Yf)
    Hf = fft2(P)

    H_conj = np.conj(Hf)
    denom = np.abs(Hf) ** 2 + K
    F_est = (H_conj / denom) * G

    y_rec = np.real(ifft2(F_est))
    y_rec = np.clip(y_rec, 0.0, 1.0)
    Y_rec = (y_rec * 255.0).astype(np.uint8)

    out_ycrcb = cv2.merge((Y_rec, Cr, Cb))
    out_bgr = cv2.cvtColor(out_ycrcb, cv2.COLOR_YCrCb2BGR)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    return out_rgb


# ================== METRICS ==================
def psnr(img1, img2):
    """
    img1, img2: uint8 RGB, same shape.
    """
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 99.0
    return 20 * np.log10(255.0 / np.sqrt(mse))


def compute_ssim(img1, img2):
    if not HAS_SSIM:
        return None
    # skimage expects grayscale or multichannel
    return float(ssim_fn(img1, img2, channel_axis=2, data_range=255))


def compute_lpips_metric(lpips_model, img1, img2, device):
    if not HAS_LPIPS or lpips_model is None:
        return None
    # img1, img2: uint8 RGB [H,W,3]
    # convert to [-1,1] tensors in NCHW
    t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t1 = (t1 * 2.0 - 1.0).to(device)
    t2 = (t2 * 2.0 - 1.0).to(device)
    with torch.no_grad():
        v = lpips_model(t1, t2)
    return float(v.item())


def tenengrad_sharpness(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag2 = gx**2 + gy**2
    return float(mag2.mean())


def laplacian_variance(gray):
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(lap.var())


# ================== CLEAN GT LOOKUP BY SUFFIX ==================
def find_clean_image_from_100k_by_suffix(blurred_basename):
    """
    For a blurred filename like:
        BLUR_L19_T81_55de4228-c37a8e3f.jpg
    extract the last block 'c37a8e3f' and look for any file in 100k
    that ends with '*c37a8e3f.jpg'.
    """
    name_no_ext, ext = os.path.splitext(blurred_basename)
    if ext == "":
        ext = ".jpg"  # fallback

    last_block = name_no_ext.split("-")[-1]  # e.g., 'c37a8e3f'
    pattern = f"*{last_block}{ext}"          # '*c37a8e3f.jpg'
    search_path = os.path.join(ORIG_100K_DIR, "**", pattern)

    matches = glob.glob(search_path, recursive=True)

    if len(matches) == 0:
        print(f"❗ No clean GT match found in 100k for suffix {last_block}{ext}")
        return None

    matches = sorted(matches, key=len)
    print(f"✔ Found ground truth in 100k (suffix match) → {matches[0]}")
    return matches[0]


# ================== YOLO HELPERS ==================
print("Loading YOLO detector (yolov8n)...")
yolo_model = YOLO("yolov8n.pt")  # will download weights first time
print("YOLO model loaded.\n")


def run_yolo_on_rgb(img_rgb, conf_thres=0.5):
    """
    img_rgb: HxWx3 uint8 RGB
    Returns: (num_dets, mean_conf_above_thr)
    """
    # YOLO can take RGB numpy directly
    results = yolo_model(img_rgb, verbose=False)[0]
    if results.boxes is None or results.boxes.conf is None:
        return 0, 0.0

    conf = results.boxes.conf.detach().cpu().numpy()
    sel = conf >= conf_thres
    if sel.sum() == 0:
        return 0, 0.0

    return int(sel.sum()), float(conf[sel].mean())


# ================== LOAD LABELS & PICK RANDOM BLURRED TEST IMAGE ==================
df = pd.read_csv(LABELS_CSV)
df_test = df[df["set"] == "test"]
df_test_blur = df_test[df_test["blur_present"] == 1].reset_index(drop=True)

if len(df_test_blur) == 0:
    raise RuntimeError("No blurred images in test set!")

row = df_test_blur.iloc[random.randint(0, len(df_test_blur) - 1)]
img_path = row["filepath"]
L_true = float(row["L"])
theta_true = float(row["theta"])
basename = os.path.basename(img_path)

print("=== RANDOM TEST SAMPLE ===")
print("Blurred path :", img_path)
print("Filename      :", basename)
print(f"True L        : {L_true:.2f}")
print(f"True θ        : {theta_true:.2f}")

# Load blurred
img_bgr_full = cv2.imread(img_path)
if img_bgr_full is None:
    raise FileNotFoundError(f"Could not read image: {img_path}")
img_rgb_full = cv2.cvtColor(img_bgr_full, cv2.COLOR_BGR2RGB)

# Find and load clean GT using suffix search
clean_path = find_clean_image_from_100k_by_suffix(basename)
if clean_path is None:
    clean_rgb = None
else:
    clean_bgr = cv2.imread(clean_path)
    clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
    # resize GT to blurred size
    if clean_rgb.shape[:2] != img_rgb_full.shape[:2]:
        clean_rgb = cv2.resize(
            clean_rgb,
            (img_rgb_full.shape[1], img_rgb_full.shape[0]),
            interpolation=cv2.INTER_AREA
        )

# ================== MODEL PREDICTION (L, θ) ==================
pil_img = Image.fromarray(img_rgb_full)
x = eval_transform(pil_img).unsqueeze(0).to(device)

with torch.no_grad():
    blur_logit, reg_out = model(x)
    blur_prob = torch.sigmoid(blur_logit).item()
    L_pred_norm = reg_out[0, 0].item()
    theta_pred_norm = reg_out[0, 1].item()

L_pred, theta_pred = denorm(
    torch.tensor(L_pred_norm),
    torch.tensor(theta_pred_norm)
)
L_pred = float(L_pred.item())
theta_pred = float(theta_pred.item())
theta_pred = max(0.0, min(180.0, theta_pred))

print("\n=== MODEL PREDICTION ===")
print(f"Blur probability      : {blur_prob:.3f}")
print(f"Predicted L           : {L_pred:.2f} px")
print(f"Predicted θ           : {theta_pred:.2f} °")
print(f"L error (pred-true)   : {L_pred - L_true:+.2f} px")
print(f"θ error (pred-true)   : {theta_pred - theta_true:+.2f} °")

# ================== DECONV USING PREDICTED PSF ==================
reg_deconv = regular_deconvolve_y_channel(img_rgb_full, L_pred, theta_pred, eps=1e-3)

# Wiener for multiple K values
wiener_results = []  # (K, img, metrics_dict)
if clean_rgb is not None:
    print("\n=== METRIC SWEEP OVER K VALUES (Wiener) ===")
else:
    print("\n=== WIENER SWEEP (no PSNR/SSIM/LPIPS vs GT because GT not found) ===")

# Setup LPIPS model if available
lpips_model = None
if HAS_LPIPS:
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()

# Precompute grayscale versions for sharpness metrics
def to_gray(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

gray_clean   = to_gray(clean_rgb)   if clean_rgb is not None else None
gray_blurred = to_gray(img_rgb_full)
gray_reg     = to_gray(reg_deconv)

for K in K_LIST:
    wimg = wiener_deconvolve_y_channel(img_rgb_full, L_pred, theta_pred, K=K)
    gray_w = to_gray(wimg)

    metrics = {
        "psnr":  None,
        "ssim":  None,
        "lpips": None,
        "tenengrad": tenengrad_sharpness(gray_w),
        "lap_var":   laplacian_variance(gray_w),
    }

    if clean_rgb is not None:
        metrics["psnr"] = psnr(clean_rgb, wimg)
        metrics["ssim"] = compute_ssim(clean_rgb, wimg) if HAS_SSIM else None
        metrics["lpips"] = compute_lpips_metric(lpips_model, clean_rgb, wimg, device) if HAS_LPIPS else None

    print(
        f"K={K:.5f} -> PSNR={metrics['psnr']:.2f} dB | "
        f"SSIM={metrics['ssim']} | LPIPS={metrics['lpips']} | "
        f"Tenengrad={metrics['tenengrad']:.1f} | LapVar={metrics['lap_var']:.1f}"
    )
    wiener_results.append((K, wimg, metrics))

# Pick best Wiener by PSNR if GT exists, else by Tenengrad (or whatever you prefer)
if clean_rgb is not None:
    best_idx = max(
        range(len(wiener_results)),
        key=lambda i: wiener_results[i][2]["psnr"]
    )
else:
    best_idx = min(
        range(len(wiener_results)),
        key=lambda i: wiener_results[i][2]["tenengrad"]  # smaller = smoother
    )

best_K, wiener_best, best_metrics = wiener_results[best_idx]

# ================== SUMMARY METRICS (BLURRED / REG / BEST WIENER) ==================
summary_rows = []

if clean_rgb is not None:
    # Blurred
    summary_rows.append({
        "variant": "Blurred",
        "K":       None,
        "psnr":    psnr(clean_rgb, img_rgb_full),
        "ssim":    compute_ssim(clean_rgb, img_rgb_full) if HAS_SSIM else None,
        "lpips":   compute_lpips_metric(lpips_model, clean_rgb, img_rgb_full, device) if HAS_LPIPS else None,
        "tenengrad": tenengrad_sharpness(gray_blurred),
        "lap_var":   laplacian_variance(gray_blurred),
    })
    # Regular
    summary_rows.append({
        "variant": "RegularDeconv",
        "K":       None,
        "psnr":    psnr(clean_rgb, reg_deconv),
        "ssim":    compute_ssim(clean_rgb, reg_deconv) if HAS_SSIM else None,
        "lpips":   compute_lpips_metric(lpips_model, clean_rgb, reg_deconv, device) if HAS_LPIPS else None,
        "tenengrad": tenengrad_sharpness(gray_reg),
        "lap_var":   laplacian_variance(gray_reg),
    })

    # Each Wiener candidate
    for K, wimg, m in wiener_results:
        summary_rows.append({
            "variant": f"Wiener_K={K:.5f}",
            "K":       K,
            "psnr":    m["psnr"],
            "ssim":    m["ssim"],
            "lpips":   m["lpips"],
            "tenengrad": m["tenengrad"],
            "lap_var":   m["lap_var"],
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n=== SUMMARY METRICS (vs CLEAN, where available) ===")
    print(summary_df)
    print("\nBest Wiener (by PSNR):")
    print(f"  K = {best_K:.5f}")
    print(f"  Metrics: {best_metrics}")
else:
    print("\n⚠ No clean GT found → summary metrics vs clean skipped.")
    summary_df = None

# ================== YOLO DETECTION METRICS ==================
print("\n=== YOLO DETECTION METRICS (COCO pretrained) ===")
det_summary = []

def add_det_row(name, img_rgb):
    n, conf = run_yolo_on_rgb(img_rgb, conf_thres=0.5)
    det_summary.append({"variant": name, "num_dets@0.5": n, "mean_conf@0.5": conf})
    print(f"{name:15s} -> #dets={n:3d}, mean_conf={conf:.3f}")

if clean_rgb is not None:
    add_det_row("Clean", clean_rgb)
add_det_row("Blurred", img_rgb_full)
add_det_row("Regular", reg_deconv)
add_det_row(f"Wiener(K={best_K:.3f})", wiener_best)

det_df = pd.DataFrame(det_summary)
print("\nYOLO detection summary:")
print(det_df)

# ================== PLOT: CLEAN | BLURRED | REGULAR | BEST WIENER ==================
plt.figure(figsize=(18, 5))

# 1) Clean GT
plt.subplot(1, 4, 1)
if clean_rgb is not None:
    plt.imshow(clean_rgb)
    title = "Clean ground truth"
else:
    plt.imshow(img_rgb_full)
    title = "Clean GT not found"
plt.title(title)
plt.axis("off")

# 2) Blurred
plt.subplot(1, 4, 2)
plt.imshow(img_rgb_full)
t2 = f"Blurred\nTrue L={L_true:.1f}, θ={theta_true:.1f}"
if clean_rgb is not None:
    t2 += f"\nPSNR vs clean = {psnr(clean_rgb, img_rgb_full):.2f} dB"
plt.title(t2)
plt.axis("off")

# 3) Regular deconv
plt.subplot(1, 4, 3)
plt.imshow(reg_deconv)
t3 = f"Regular deconv\nPred L={L_pred:.1f}, θ={theta_pred:.1f}"
if clean_rgb is not None:
    t3 += f"\nPSNR vs clean = {psnr(clean_rgb, reg_deconv):.2f} dB"
plt.title(t3)
plt.axis("off")

# 4) Best Wiener deconv
plt.subplot(1, 4, 4)
plt.imshow(wiener_best)
t4 = f"Wiener (best K={best_K:.5f})\nPred L={L_pred:.1f}, θ={theta_pred:.1f}"
if clean_rgb is not None:
    t4 += f"\nPSNR vs clean = {best_metrics['psnr']:.2f} dB"
plt.title(t4)
plt.axis("off")

plt.tight_layout()
plt.show()
