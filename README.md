# Explainable Deep Learning for Early Alzheimer’s Detection 🧠

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Computer_Vision-Model-blue?style=for-the-badge)
![Research](https://img.shields.io/badge/Research-Grad--CAM-success?style=for-the-badge)

## 📌 The Goal
This project implements a **Convolutional Neural Network (CNN)** to detect Alzheimer's Dementia from 3D MRI scans. Beyond simple classification, it utilizes **Grad-CAM (Gradient-weighted Class Activation Mapping)** to provide explainability, highlighting exactly *where* in the brain the model detects anomalies.

## 📸 Model Interpretation (The "Why")
The image below demonstrates the model's decision-making process. The **Red/Yellow** regions indicate areas of high activation—specifically focusing on cortical atrophy and ventricular enlargement key to Alzheimer's diagnosis.

![Model Explanation](gradcam_result.jpg)
*(Figure 1: Side-by-side comparison of raw MRI input vs. Grad-CAM activation heatmap.)*

---

## 🛠️ Technical Stack
* **Core Framework:** PyTorch & Torchvision
* **Architecture:** Custom 3-Layer CNN with Max Pooling
* **Explainability:** `pytorch-grad-cam`
* **Data Processing:** OpenCV & NumPy

## 📂 Project Structure
* `dataset.py`: ETL pipeline (Image loading, resizing to 128x128, Normalization).
* `model.py`: PyTorch CNN definition.
* `train.py`: Training loop with CrossEntropyLoss and Adam Optimizer.
* `explain.py`: Visualization script using Grad-CAM.

## 🚀 How to Run
1.  **Install Dependencies:**
    ```bash
    pip install torch torchvision opencv-python matplotlib grad-cam
    ```
2.  **Train the Model:**
    ```bash
    python train.py
    ```
    *(Achieved Loss < 0.10 after 10 Epochs)*
3.  **Generate Explanation:**
    ```bash
    python explain.py
    ```