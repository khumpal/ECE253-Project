import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ==========================
# CONFIG
# ==========================
BASE_DIR    = r"C:\ECE253Project"
DATA_DIR    = os.path.join(BASE_DIR, "multi_degraded_100k")
LABELS_CSV  = os.path.join(DATA_DIR, "labels_multitask.csv")

IMG_SIZE    = 224
BATCH_SIZE  = 128
NUM_EPOCHS  = 20
LR          = 1e-4

# LR scheduler (optional but helpful)
STEP_SIZE   = 8       # drop LR every 8 epochs
GAMMA       = 0.1     # new_lr = old_lr * GAMMA

# Normalization ranges for regression targets
L_MIN, L_MAX         = 0.0, 60.0
THETA_MIN, THETA_MAX = 0.0, 180.0
THETA_RANGE          = THETA_MAX - THETA_MIN  # 180

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ==========================
# DATASET
# ==========================
class MultiDegradeDataset(Dataset):
    """
    Expects a DataFrame with columns:
      filepath, blur_present, fog_present, lowlight_present, L, theta, set, ...
    """
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["filepath"]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        blur_present     = float(row["blur_present"])
        fog_present      = float(row["fog_present"])
        lowlight_present = float(row["lowlight_present"])
        L                = float(row["L"])
        theta            = float(row["theta"])

        # Normalize L and theta into [0,1] based on the configured ranges
        L_norm = (L - L_MIN) / (L_MAX - L_MIN) if (L_MAX > L_MIN) else 0.0
        theta_norm = (theta - THETA_MIN) / (THETA_MAX - THETA_MIN) if (THETA_MAX > THETA_MIN) else 0.0

        # Clamp to [0,1] to be safe
        L_norm = max(0.0, min(1.0, L_norm))
        theta_norm = max(0.0, min(1.0, theta_norm))

        target = {
            # 3 binary tasks
            "blur_present":     torch.tensor(blur_present,     dtype=torch.float32),
            "fog_present":      torch.tensor(fog_present,      dtype=torch.float32),
            "lowlight_present": torch.tensor(lowlight_present, dtype=torch.float32),
            # regression (normalized)
            "L_norm":           torch.tensor(L_norm,           dtype=torch.float32),
            "theta_norm":       torch.tensor(theta_norm,       dtype=torch.float32),
        }
        return img, target


# ==========================
# MODEL: RESNET18 MULTI-TASK
# ==========================
class MultiTaskResNet(nn.Module):
    """
    Backbone: ResNet-18
    Heads:
      - cls_head: 3 logits (blur, fog, lowlight)
      - reg_head: 2 outputs (L_norm, theta_norm)
    """
    def __init__(self, backbone, num_feats):
        super().__init__()
        self.backbone = backbone
        self.backbone.fc = nn.Identity()

        # classification head: [B, num_feats] -> [B, 3]
        self.cls_head = nn.Linear(num_feats, 3)

        # regression head: [B, num_feats] -> [B, 2]
        self.reg_head = nn.Sequential(
            nn.Linear(num_feats, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # [L_norm, theta_norm]
        )

    def forward(self, x):
        feats = self.backbone(x)
        cls_logits = self.cls_head(feats)     # [B, 3]
        reg_out    = self.reg_head(feats)     # [B, 2]
        return cls_logits, reg_out


def denorm(L_norm, theta_norm):
    """
    Convert normalized L, theta back to:
      L in [L_MIN, L_MAX]
      theta in [THETA_MIN, THETA_MAX]
    """
    L = L_norm * (L_MAX - L_MIN) + L_MIN
    theta = theta_norm * (THETA_MAX - THETA_MIN) + THETA_MIN
    return L, theta


# ==========================
# CIRCULAR THETA LOSS
# ==========================
def circular_theta_loss(theta_pred_norm, theta_true_norm):
    """
    theta_pred_norm, theta_true_norm: tensors of shape [N] in [0,1]

    We interpret them as angles in [0, 180] degrees and use a circular loss:

      diff_deg = (theta_pred_deg - theta_true_deg)  (in degrees)
      wrapped  = ((diff_deg + 90) % 180) - 90

    Then we apply MSE on (wrapped / 180) to keep it normalized.
    """
    # Convert normalized to degrees
    diff_norm = theta_pred_norm - theta_true_norm          # in normalized units
    diff_deg  = diff_norm * THETA_RANGE                    # -> degrees

    # Wrap to [-90, +90]
    wrapped = (diff_deg + 90.0) % 180.0 - 90.0             # still differentiable in PyTorch

    # Normalize back to [-0.5,0.5] approx, then MSE
    wrapped_norm = wrapped / THETA_RANGE                   # scale ~[-0.5, 0.5]
    loss = (wrapped_norm ** 2).mean()
    return loss


# ==========================
# EVALUATION
# ==========================
def evaluate(model, loader, device):
    """
    Returns:
      blur_acc, fog_acc, lowlight_acc, L_mae, theta_mae_circ
      (MAEs computed only on samples with blur_present == 1)
    """
    model.eval()

    total_blur_correct = 0
    total_fog_correct = 0
    total_low_correct = 0
    total_samples = 0

    total_L_err = 0.0
    total_theta_err = 0.0
    total_blurred = 0

    with torch.no_grad():
        for imgs, target in loader:
            imgs = imgs.to(device, non_blocking=True)

            blur_true = target["blur_present"].to(device, non_blocking=True)
            fog_true = target["fog_present"].to(device, non_blocking=True)
            low_true = target["lowlight_present"].to(device, non_blocking=True)
            L_true_norm = target["L_norm"].to(device, non_blocking=True)
            theta_true_norm = target["theta_norm"].to(device, non_blocking=True)

            cls_logits, reg_out = model(imgs)

            # classification predictions
            cls_prob = torch.sigmoid(cls_logits)  # [B,3]
            cls_pred = (cls_prob > 0.5).float()   # [B,3]

            blur_pred = cls_pred[:, 0]
            fog_pred  = cls_pred[:, 1]
            low_pred  = cls_pred[:, 2]

            total_blur_correct += (blur_pred == blur_true).float().sum().item()
            total_fog_correct  += (fog_pred  == fog_true ).float().sum().item()
            total_low_correct  += (low_pred  == low_true ).float().sum().item()
            total_samples      += blur_true.numel()

            # regression metrics only on blurred samples
            mask = blur_true > 0.5
            if mask.sum() > 0:
                L_pred_norm     = reg_out[:, 0][mask]
                theta_pred_norm = reg_out[:, 1][mask]
                L_true_norm_m   = L_true_norm[mask]
                theta_true_norm_m = theta_true_norm[mask]

                # denormalize
                L_pred, theta_pred = denorm(L_pred_norm, theta_pred_norm)
                L_true, theta_true = denorm(L_true_norm_m, theta_true_norm_m)

                # clamp theta to [0,180]
                theta_pred = torch.clamp(theta_pred, THETA_MIN, THETA_MAX)
                theta_true = torch.clamp(theta_true, THETA_MIN, THETA_MAX)

                # L MAE
                total_L_err += torch.abs(L_pred - L_true).sum().item()

                # circular θ MAE
                diff = torch.abs(theta_pred - theta_true)
                circ_diff = torch.minimum(diff, THETA_RANGE - diff)  # min(|Δθ|, 180-|Δθ|)
                total_theta_err += circ_diff.sum().item()

                total_blurred += mask.sum().item()

    blur_acc = total_blur_correct / total_samples if total_samples > 0 else 0.0
    fog_acc  = total_fog_correct  / total_samples if total_samples > 0 else 0.0
    low_acc  = total_low_correct  / total_samples if total_samples > 0 else 0.0

    L_mae = total_L_err / total_blurred if total_blurred > 0 else 0.0
    theta_mae = total_theta_err / total_blurred if total_blurred > 0 else 0.0

    return blur_acc, fog_acc, low_acc, L_mae, theta_mae


# ==========================
# MAIN TRAINING
# ==========================
def main():
    print("Loading labels from:", LABELS_CSV)
    df = pd.read_csv(LABELS_CSV)

    df_train = df[df["set"] == "train"]
    df_val   = df[df["set"] == "val"]
    df_test  = df[df["set"] == "test"]

    print(f"Dataset sizes → Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # Optional sanity checks
    print("\nTrain blur_present counts:\n", df_train["blur_present"].value_counts())
    print("Train fog_present counts:\n", df_train["fog_present"].value_counts())
    print("Train lowlight_present counts:\n", df_train["lowlight_present"].value_counts())

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Datasets & loaders
    train_ds = MultiDegradeDataset(df_train, train_transform)
    val_ds   = MultiDegradeDataset(df_val,   eval_transform)
    test_ds  = MultiDegradeDataset(df_test,  eval_transform)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True
    )

    print(f"Train steps/epoch: {len(train_loader)}")
    print(f"Val   steps/epoch: {len(val_loader)}")
    print(f"Test  steps/epoch: {len(test_loader)}")

    # Model
    print("\nInitializing model...")
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_feats = backbone.fc.in_features
    model = MultiTaskResNet(backbone, num_feats).to(device)

    # Losses
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()  # for L_norm

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    ckpt_dir = os.path.join(BASE_DIR, "checkpoints_multitask")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ======================
    # Training loop
    # ======================
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n===== Epoch {epoch}/{NUM_EPOCHS} | LR={current_lr:.6f} =====")

        for batch_idx, (imgs, target) in enumerate(train_loader, start=1):
            imgs = imgs.to(device, non_blocking=True)

            blur_true = target["blur_present"].to(device, non_blocking=True)
            fog_true  = target["fog_present"].to(device, non_blocking=True)
            low_true  = target["lowlight_present"].to(device, non_blocking=True)
            L_true_norm = target["L_norm"].to(device, non_blocking=True)
            theta_true_norm = target["theta_norm"].to(device, non_blocking=True)

            cls_logits, reg_out = model(imgs)
            L_pred_norm     = reg_out[:, 0]
            theta_pred_norm = reg_out[:, 1]

            # --- Classification loss (3 tasks) ---
            cls_targets = torch.stack([blur_true, fog_true, low_true], dim=1)  # [B,3]
            loss_cls = bce_loss(cls_logits, cls_targets)

            # --- Regression loss (only when blur_present == 1) ---
            mask = blur_true > 0.5
            if mask.sum() > 0:
                # L MSE in normalized space
                loss_L = mse_loss(L_pred_norm[mask], L_true_norm[mask])

                # circular θ loss in normalized space
                loss_theta = circular_theta_loss(
                    theta_pred_norm[mask],
                    theta_true_norm[mask]
                )
                loss_reg = loss_L + loss_theta
            else:
                loss_reg = torch.tensor(0.0, device=device)

            # combine
            loss = loss_cls + loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0 or batch_idx == 1 or batch_idx == len(train_loader):
                avg_so_far = running_loss / batch_idx
                print(
                    f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} "
                    f"| BatchLoss={loss.item():.4f} | AvgLoss={avg_so_far:.4f}"
                )

        train_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch {epoch} finished. Computing validation metrics...")

        blur_acc_val, fog_acc_val, low_acc_val, L_mae_val, theta_mae_val = evaluate(model, val_loader, device)
        print(
            f"[Epoch {epoch}] TrainLoss={train_loss:.4f}  "
            f"ValBlurAcc={blur_acc_val:.3f}  "
            f"ValFogAcc={fog_acc_val:.3f}   "
            f"ValLowlightAcc={low_acc_val:.3f}  "
            f"Val L MAE={L_mae_val:.2f}px  "
            f"Val θ MAE (circ)={theta_mae_val:.2f}°"
        )

        # LR schedule
        scheduler.step()

        # Save checkpoint
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"💾 Checkpoint saved to {ckpt_path}")

    # ======================
    # Final test evaluation
    # ======================
    print("\nRunning final TEST evaluation...")
    blur_acc_t, fog_acc_t, low_acc_t, L_mae_t, theta_mae_t = evaluate(model, test_loader, device)
    print("\n=== FINAL TEST RESULTS ===")
    print(f"Blur presence accuracy   : {blur_acc_t:.3f}")
    print(f"Fog presence accuracy    : {fog_acc_t:.3f}")
    print(f"Lowlight presence acc    : {low_acc_t:.3f}")
    print(f"L MAE on blurred imgs    : {L_mae_t:.2f} px")
    print(f"θ MAE (circular) on blur : {theta_mae_t:.2f} °")

    # Save final model
    model_path = os.path.join(BASE_DIR, "multitask_resnet18_blur_fog_lowlight.pth")
    torch.save(model.state_dict(), model_path)
    print("\n✅ Final model saved to:", model_path)


if __name__ == "__main__":
    main()
