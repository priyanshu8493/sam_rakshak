import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image # PyTorch transforms work best with PIL images

# --- Constants ---
# DINOv2 was trained on 224x224 images. We must use this size.
IMG_SIZE = 224

# --- DINOv2 specific normalization ---
# These are the mean/std values from the ImageNet dataset
DINO_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def preprocess_frame_dino(frame):
    """
    Converts a single OpenCV video frame (BGR) into a DINOv2-ready tensor.
    
    1. Converts BGR (OpenCV) to RGB (PIL)
    2. Applies the DINOv2 transforms (resize, ToTensor, normalize)
    3. Adds a batch dimension (1, C, H, W)
    
    Args:
        frame (numpy.ndarray): The raw video frame from OpenCV.
        
    Returns:
        torch.Tensor: The processed tensor, ready for the model.
    """
    try:
        # 1. Convert BGR (OpenCV) to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Convert numpy array to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # 3. Apply DINOv2 transforms
        tensor = DINO_TRANSFORM(pil_image)
        
        # 4. Add batch dimension (C, H, W) -> (1, C, H, W)
        return tensor.unsqueeze(0)
    
    except Exception as e:
        print(f"[DataHandler] Error processing frame: {e}")
        return None

if __name__ == "__main__":
    """
    A simple test script to show the preprocessor in action.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
        
    print("Opening webcam... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        tensor = preprocess_frame_dino(frame)
        
        if tensor is not None:
            # To display, we need to un-normalize (this is complex)
            # For a simple check, just show the original
            cv2.imshow("Original Feed", frame)
            # And print the tensor shape
            # print(f"Output tensor shape: {tensor.shape}") # Uncomment for debugging
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

