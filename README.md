**Multi-Degradation Image Restoration Pipeline

ECE 253 Final Project**

This repository contains an end-to-end image restoration system for detecting and correcting motion blur, fog, and low-light degradation, including stacked combinations of these effects. The system combines a multi-task CNN (ResNet-18) with classical image restoration techniques and compares a smart, gated pipeline against a naive always-on baseline.
**
Project Overview**

Real-world images often suffer from multiple degradations simultaneously (e.g., blur + fog + low-light). Traditional pipelines either over-process images or fail when degradations are stacked.
Our approach:

1) Trains a single multi-task CNN to jointly detect blur, fog, and low-light
2) Regresses motion blur parameters (length and angle)
3) Uses predictions to selectively activate restoration modules
4) Avoids unnecessary processing on already clean images
5) Demonstrates improved perceptual quality and downstream detection performance

Repository Structure
.
├── multi_degraded_tiny/                     # Tiny evaluation dataset (~10–20 images)
│   ├── train/
│   ├── val/
│   ├── test/
│   └── labels_multitask_tiny.csv
│
├── dataset_generator_full.py                # Full dataset generator (BDD100K)
├── dataset_generator_tiny.py                # Tiny dataset generator (GitHub-friendly)
├── full_cnn.py                              # CNN training script
├── pipeline.py                              # Smart vs naive restoration pipeline
├── multitask_resnet18_blur_fog_lowlight.pth # Pretrained CNN model
├── yolov8n.pt                               # YOLOv8 model for detection metrics
└── README.md

**Pretrained Model**
A fully trained multi-task ResNet-18 model is included:

multitask_resnet18_blur_fog_lowlight.pth

This model was trained on ~400,000 synthetically degraded images generated from the BDD100K dataset.
No training is required to run the pipeline.

**Tiny Dataset (Included)**
The folder multi_degraded_tiny/ contains a small synthetic dataset so the pipeline can be:

Run immediately
Evaluated end-to-end
Stored in GitHub without large files
This dataset mirrors the structure of the full dataset and includes clean images, degraded images, and a CSV label file.

**Full Dataset (Optional)**

The full dataset is generated from BDD100K, which is too large to include in this repository.

🔗 BDD100K Dataset:
https://bdd-data.berkeley.edu/

To reproduce full training:
Download BDD100K images and labels
Update paths in dataset_generator_full.py
Generate the dataset
Train using full_cnn.py

**Dependencies**
Install Python 3.8+ and run:

pip install torch torchvision
pip install opencv-python numpy pandas matplotlib pillow
pip install scikit-image
pip install lpips
pip install ultralytics

For GPU support, install a CUDA-enabled version of PyTorch.

**How to Run the Pipeline**
1. Update Base Directory

Open pipeline.py and modify this line: (LINE 35)

BASE_DIR = r"C:\path\to\your\project\folder"  # update as needed 
Set it to the directory where this repository is located.

**TO RUN**

From the project directory:

python pipeline.py

The script will:

1) Load the pretrained CNN
2) Run a demo comparison (smart vs naive pipeline)
3) Evaluate on the tiny dataset
   
Compute metrics:
PSNR
SSIM
LPIPS (no-reference)
YOLOv8 detection count and confidence
Generate qualitative and quantitative plots

Evaluation Notes

Smart pipeline applies restoration only when degradations are confidently detected
Naive pipeline always applies all restoration modules
LPIPS is used without ground truth by comparing outputs to the degraded input
YOLOv8 metrics show impact on downstream perception tasks
The smart pipeline improves visibility while avoiding over-enhancement artifacts

Custom Real-World Images
The pipeline also supports naturally degraded real images with no ground truth.
In this setting:
LPIPS measures perceptual change relative to the input
YOLO metrics quantify semantic improvement
Results demonstrate robustness beyond synthetic data

**Acknowledgements
**

BDD100K Dataset
LPIPS (Zhang et al.)
YOLOv8 (Ultralytics)
ResNet (He et al.)


