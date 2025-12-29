import torch
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from model import AlzheimerCNN
import matplotlib.pyplot as plt
import os

def generate_heatmap(image_path, model_path="best_model.pth"):
    # 1. Load Model
    device = torch.device("cpu") # Visualization is safer on CPU
    model = AlzheimerCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    target_layers = [model.conv3]

    # 2. Prepare Image
    img_raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_raw is None:
        raise FileNotFoundError(f"Could not open image at {image_path}")
        
    img_raw = cv2.resize(img_raw, (128, 128))
    input_tensor = torch.tensor(img_raw / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    rgb_img = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB)
    rgb_img = np.float32(rgb_img) / 255

    # 3. Initialize Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)

    # 4. Generate Heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # 5. Display Result
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title("Original MRI Scan")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title("AI Focus (Grad-CAM)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    
    cv2.imwrite("gradcam_result.jpg", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR) * 255)
    print("Result saved as 'gradcam_result.jpg'")

if __name__ == "__main__":
    # Point this to a real image in your dataset
    # You can change 'NonDemented' to 'Demented' to test different brains
    test_image_path = "./Dataset/Demented/scan_101.jpg" 
    
    # Small helper to find ANY image if scan_101 doesn't exist
    if not os.path.exists(test_image_path):
        # Just grab the first file in the directory
        folder = "./Dataset/Demented"
        files = os.listdir(folder)
        if len(files) > 0:
            test_image_path = os.path.join(folder, files[0])
    
    if os.path.exists(test_image_path):
        print(f"Explaining image: {test_image_path}")
        generate_heatmap(test_image_path)
    else:
        print("Could not find an image to test. Check your Dataset folder.")