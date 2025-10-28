import cv2
import numpy as np
import matplotlib.pyplot as plt

def motion_blur(img, L=15, theta=45):
    """
    Apply linear motion blur of length L (pixels) at angle theta (degrees).
    """
    # Ensure odd kernel size
    ksize = int(L)
    if ksize % 2 == 0:
        ksize += 1

    # Create an empty kernel
    kernel = np.zeros((ksize, ksize))

    # Fill the center row with ones (horizontal line)
    kernel[ksize // 2, :] = np.ones(ksize)

    # Normalize so sum = 1
    kernel = kernel / ksize

    # Rotate kernel by theta degrees
    M = cv2.getRotationMatrix2D((ksize / 2 - 0.5, ksize / 2 - 0.5), theta, 1)
    kernel = cv2.warpAffine(kernel, M, (ksize, ksize))

    # Apply filter
    blurred = cv2.filter2D(img, -1, kernel)

    return blurred, kernel

# ---- Load and apply blur ----
img = cv2.imread("IMG_6687.JPG")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

L = 50       # motion blur length (pixels)
theta = 180   # motion direction (degrees)

blurred, kernel = motion_blur(img, L, theta)

# ---- Show results ----
plt.figure(figsize=(12,5))
plt.subplot(1,3,1); plt.imshow(img); plt.title("Original"); plt.axis('off')
plt.subplot(1,3,2); plt.imshow(kernel, cmap='gray'); plt.title(f"Motion Kernel (L={L}, θ={theta}°)")
plt.subplot(1,3,3); plt.imshow(blurred); plt.title("Blurred"); plt.axis('off')
plt.tight_layout()
plt.show()
