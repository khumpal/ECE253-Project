import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ====== CONFIG ======
BASE_DIR    = r"C:\ECE253Project"  # update if needed
DATA_DIR    = os.path.join(BASE_DIR, "blurred_50_50")
LABELS_CSV  = os.path.join(DATA_DIR, "labels.csv")

IMG_SIZE    = 224
BATCH_SIZE  = 128        # tune if you hit OOM
NUM_EPOCHS  = 20
LR          = 1e-4

# LR scheduler settings
STEP_SIZE   = 8          # drop LR every 8 epochs
GAMMA       = 0.1        # new_lr = old_lr * GAMMA

L_MIN, L_MAX = 0.0, 60.0
THETA_MIN, THETA_MAX = 0.0, 180.0   # we work in [0, 180]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ====== DATASET ======
class BlurDataset(Dataset):
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

        blur_present = float(row["blur_present"])
        L = float(row["L"])
        theta = float(row["theta"])

        # normalize L and theta for regression
        L_norm = (L - L_MIN) / (L_MAX - L_MIN) if (L_MAX > L_MIN) else 0.0
        theta_norm = (theta - THETA_MIN) / (THETA_MAX - THETA_MIN) if (THETA_MAX > THETA_MIN) else 0.0

        target = {
            "blur_present": torch.tensor(blur_present, dtype=torch.float32),
            "L_norm": torch.tensor(L_norm, dtype=torch.float32),
            "theta_norm": torch.tensor(theta_norm, dtype=torch.float32)
        }
        return img, target


# ====== MODEL: RESNET18 MULTI-TASK ======
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
        reg_out = self.reg_head(feats)  # [B, 2]
        return blur_logit, reg_out


# ====== DENORM HELPERS ======
def denorm(L_norm, theta_norm):
    L = L_norm * (L_MAX - L_MIN) + L_MIN
    theta = theta_norm * (THETA_MAX - THETA_MIN) + THETA_MIN
    return L, theta


# ====== EVALUATION (with circular θ error) ======
def evaluate(model, loader, device):
    """
    Returns:
      blur_acc   : accuracy of blur_present (0/1)
      L_mae      : MAE of blur length L (pixels) on blurred samples
      theta_mae  : *circular* MAE of θ (degrees) on blurred samples
                   using min(|Δθ|, 180 - |Δθ|).
    """
    model.eval()
    total_blur_correct = 0
    total_samples = 0

    total_L_err = 0.0
    total_theta_err = 0.0
    total_blurred = 0

    with torch.no_grad():
        for imgs, target in loader:
            imgs = imgs.to(device, non_blocking=True)
            blur_true = target["blur_present"].to(device, non_blocking=True)
            L_true_norm = target["L_norm"].to(device, non_blocking=True)
            theta_true_norm = target["theta_norm"].to(device, non_blocking=True)

            blur_logit, reg_out = model(imgs)
            blur_prob = torch.sigmoid(blur_logit)
            blur_pred = (blur_prob > 0.5).float()

            total_blur_correct += (blur_pred == blur_true).float().sum().item()
            total_samples += blur_true.numel()

            # regression metrics only on blurred samples
            mask = blur_true > 0.5
            if mask.sum() > 0:
                L_pred_norm = reg_out[:, 0][mask]
                theta_pred_norm = reg_out[:, 1][mask]
                L_true_norm_m = L_true_norm[mask]
                theta_true_norm_m = theta_true_norm[mask]

                # de-normalize
                L_pred, theta_pred = denorm(L_pred_norm, theta_pred_norm)
                L_true, theta_true = denorm(L_true_norm_m, theta_true_norm_m)

                # clamp predicted theta into [0, 180] just to be safe
                theta_pred = torch.clamp(theta_pred, 0.0, 180.0)
                theta_true = torch.clamp(theta_true, 0.0, 180.0)

                # L MAE (straight)
                total_L_err += torch.abs(L_pred - L_true).sum().item()

                # θ MAE using *circular* difference on [0, 180]
                diff = torch.abs(theta_pred - theta_true)          # |Δθ|
                circ_diff = torch.minimum(diff, 180.0 - diff)      # min(|Δθ|, 180-|Δθ|)
                total_theta_err += circ_diff.sum().item()

                total_blurred += mask.sum().item()

    blur_acc = total_blur_correct / total_samples if total_samples > 0 else 0.0
    L_mae = total_L_err / total_blurred if total_blurred > 0 else 0.0
    theta_mae = total_theta_err / total_blurred if total_blurred > 0 else 0.0
    return blur_acc, L_mae, theta_mae


def main():
    print("Loading labels from:", LABELS_CSV)
    df = pd.read_csv(LABELS_CSV)
    df_train = df[df["set"] == "train"]
    df_val   = df[df["set"] == "val"]
    df_test  = df[df["set"] == "test"]

    print(f"Dataset sizes -> Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # Optional: sanity check blur balance
    print("\nTrain blur_present counts:\n", df_train["blur_present"].value_counts())
    print("Val blur_present counts:\n", df_val["blur_present"].value_counts())
    print("Test blur_present counts:\n", df_test["blur_present"].value_counts())

    # ====== TRANSFORMS ======
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])

    train_ds = BlurDataset(df_train, train_transform)
    val_ds   = BlurDataset(df_val,   eval_transform)
    test_ds  = BlurDataset(df_test,  eval_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,          # keep 0 on Windows to avoid multiprocessing issues
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"Train steps per epoch: {len(train_loader)}")
    print(f"Val   steps per epoch: {len(val_loader)}")
    print(f"Test  steps per epoch: {len(test_loader)}")

    # ====== MODEL ======
    print("Initializing model...")
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_feats = resnet.fc.in_features
    model = MultiTaskResNet(resnet, num_feats).to(device)

    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ====== LR SCHEDULER (StepLR) ======
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=STEP_SIZE, gamma=GAMMA
    )

    # ====== TRAINING LOOP ======
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        # print current LR
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n===== Epoch {epoch}/{NUM_EPOCHS} | LR={current_lr:.6f} =====")

        for batch_idx, (imgs, target) in enumerate(train_loader, start=1):
            imgs = imgs.to(device, non_blocking=True)
            blur_true = target["blur_present"].to(device, non_blocking=True)
            L_true_norm = target["L_norm"].to(device, non_blocking=True)
            theta_true_norm = target["theta_norm"].to(device, non_blocking=True)

            blur_logit, reg_out = model(imgs)
            L_pred_norm = reg_out[:, 0]
            theta_pred_norm = reg_out[:, 1]

            # classification loss
            loss_blur = bce_loss(blur_logit, blur_true)

            # regression loss only for blurred images
            mask = blur_true > 0.5
            if mask.sum() > 0:
                loss_L = mse_loss(L_pred_norm[mask], L_true_norm[mask])
                loss_theta = mse_loss(theta_pred_norm[mask], theta_true_norm[mask])
                loss_reg = loss_L + loss_theta
            else:
                loss_reg = torch.tensor(0.0, device=device)

            loss = loss_blur + loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Progress print every N batches
            if batch_idx % 100 == 0 or batch_idx == 1 or batch_idx == len(train_loader):
                avg_so_far = running_loss / batch_idx
                print(
                    f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} "
                    f"| BatchLoss={loss.item():.4f} | AvgLoss={avg_so_far:.4f}"
                )

        train_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch {epoch} finished. Computing validation metrics...")
        blur_acc_val, L_mae_val, theta_mae_val = evaluate(model, val_loader, device)
        print(
            f"[Epoch {epoch}] TrainLoss={train_loss:.4f}  "
            f"ValBlurAcc={blur_acc_val:.3f}  "
            f"Val L MAE={L_mae_val:.2f}px  "
            f"Val θ MAE (circular)={theta_mae_val:.2f}°"
        )

        # Step the LR scheduler AFTER validation
        scheduler.step()

        # ===== AUTO-SAVE CHECKPOINT AFTER EACH EPOCH =====
        ckpt_dir = os.path.join(BASE_DIR, "checkpoints_circ_stepLR")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"💾 Checkpoint saved to {ckpt_path}")

    # ====== FINAL TEST EVAL ======
    print("\nRunning final test evaluation...")
    blur_acc_test, L_mae_test, theta_mae_test = evaluate(model, test_loader, device)
    print("\n=== TEST RESULTS (circular θ error) ===")
    print(f"Blur presence accuracy : {blur_acc_test:.3f}")
    print(f"L MAE on blurred imgs  : {L_mae_test:.2f} px")
    print(f"θ MAE on blurred imgs  : {theta_mae_test:.2f} °")

    # NEW MODEL NAME
    model_path = os.path.join(BASE_DIR, "blur_multitask_resnet18_circ_stepLR.pth")
    torch.save(model.state_dict(), model_path)
    print("Model saved to:", model_path)


if __name__ == "__main__":
    main()
