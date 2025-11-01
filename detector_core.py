import torch
import numpy as np
import time
import cv2
from sklearn.neighbors import NearestNeighbors
# We must import from the user's file on the right
from data_handler import preprocess_frame_dino, IMG_SIZE

# --- Constants ---
# DINOv2 ViT-Small/14 patch size is 14. 224 / 14 = 16.
# So our feature map will be 16x16 patches.
PATCH_GRID_SIZE = 16 
# Number of patches (16*16)
NUM_PATCHES = PATCH_GRID_SIZE * PATCH_GRID_SIZE 
# Number of "normal" features to store in our memory bank.
# A larger number is more accurate but slower. 1024 is a good start.
MEMORY_BANK_SIZE = 1024
# How many standard deviations above the mean to set the threshold
THRESHOLD_STRICTNESS = 3.5 

class AnomalyDetector:
    """
    A backend-ready class to handle anomaly detection using DINOv2
    and a k-NN Memory Bank.
    """
    
    def __init__(self):
        print("[DetectorCore] Initializing...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DetectorCore] Using device: {self.device}")
        
        self.model = None
        self.threshold = None
        self.is_calibrated = False
        self.memory_bank = None
        # k-NN model should find the *single* nearest neighbor
        self.nn_model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')

        try:
            # This will download the model (dinov2_vits14) the first time.
            # Total size is ~340MB.
            print("[DetectorCore] Loading DINOv2 model (dinov2_vits14)...")
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.model.to(self.device)
            self.model.eval() # Set to evaluation mode
            print("[DetectorCore] DINOv2 model loaded.")
        except Exception as e:
            print(f"[DetectorCore] Error loading DINOv2 model: {e}")
            print("Please ensure you have an internet connection for the first run.")

    def calibrate(self, cap, calibration_time_sec=60):
        """
        Calibrates the anomaly threshold by building the memory bank.
        This is a generator function that yields frames for the web UI.
        
        Args:
            cap (cv2.VideoCapture): The active video capture object.
            calibration_time_sec (int): How long to run calibration.
        """
        if not self.model:
            print("[DetectorCore] Error: Model is not loaded. Cannot calibrate.")
            return

        print(f"[DetectorCore] Starting calibration for {calibration_time_sec}s...")
        print("[DetectorCore] Please show ONLY 'normal' activity.")
        
        all_normal_features = []
        start_time = time.time()
        
        while time.time() - start_time < calibration_time_sec:
            ret, frame = cap.read() # Read frame from the *existing* capture
            if not ret:
                break
            
            tensor = preprocess_frame_dino(frame)
            if tensor is None:
                continue
                
            tensor = tensor.to(self.device)
            
            # Get patch features from DINOv2
            with torch.no_grad():
                features_dict = self.model.forward_features(tensor)
                # DINOv2 returns a dict, we want the patch tokens
                # Shape: (1, 256, 384) -> (NumPatches, FeatureDim)
                patch_features = features_dict['x_norm_patchtokens'].cpu().numpy()[0]
            
            all_normal_features.append(patch_features)
            
            # We must yield a frame to keep the stream alive
            remaining = int(calibration_time_sec - (time.time() - start_time))
            text = f"CALIBRATING... ({remaining}s)"
            cv2.putText(frame, text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "Show 'normal' activity", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            yield frame

        if not all_normal_features:
            print("[DetectorCore] Calibration FAILED. No frames read.")
            self.is_calibrated = False
            return

        # --- Build the Memory Bank ---
        # Concatenate all features into one giant array
        all_normal_features = np.concatenate(all_normal_features, axis=0)
        
        # Coreset Subsampling: Select a random subset to be our memory bank
        if len(all_normal_features) > MEMORY_BANK_SIZE:
            indices = np.random.choice(len(all_normal_features), MEMORY_BANK_SIZE, replace=False)
            self.memory_bank = all_normal_features[indices]
        else:
            self.memory_bank = all_normal_features
            
        print(f"[DetectorCore] Building k-NN index from {len(self.memory_bank)} features...")
        
        # Build the NearestNeighbors model (k=1)
        self.nn_model.fit(self.memory_bank)
        
        # --- Set Threshold ---
        print("[DetectorCore] Calculating threshold...")
        
        # --- THIS IS THE FIX ---
        # Calculate distances for *ALL* normal features, not just the memory bank.
        # This prevents the threshold from being 0.0.
        distances, _ = self.nn_model.kneighbors(all_normal_features)
        # --- END FIX ---
        
        # Set threshold to be mean + N standard deviations of the normal scores
        self.threshold = np.mean(distances) + THRESHOLD_STRICTNESS * np.std(distances)
        self.is_calibrated = True
        print(f"[DetectorCore] Calibration complete. Threshold set to: {self.threshold:.4f}")

    def process_frame(self, frame):
        """
        Processes a single frame and returns its anomaly status and heatmap.
        """
        if not self.is_calibrated:
            return {"error": "Detector not calibrated. Please wait..."}

        tensor = preprocess_frame_dino(frame)
        if tensor is None:
            return {"error": "Invalid frame."}
            
        tensor = tensor.to(self.device)

        # 1. Get patch features
        with torch.no_grad():
            features_dict = self.model.forward_features(tensor)
            patch_features = features_dict['x_norm_patchtokens'].cpu().numpy()[0] # Shape (256, 384)

        # 2. Find nearest neighbor distance for EACH patch
        # distances shape: (256, 1)
        distances, _ = self.nn_model.kneighbors(patch_features)
        
        # 3. Create the heatmap (16x16 grid of scores)
        heatmap = distances.reshape((PATCH_GRID_SIZE, PATCH_GRID_SIZE))
        
        # 4. Get global score (the most anomalous patch)
        global_score = np.max(heatmap)
        
        is_anomaly = global_score > self.threshold
        
        return {
            "global_score": global_score,
            "is_anomaly": is_anomaly,
            "heatmap": heatmap # This is the 16x16 heatmap
        }

if __name__ == "__main__":
    """
    A simple test script to show how to use the DINOv2 AnomalyDetector.
    """
    detector = AnomalyDetector()
    if not detector.model:
        exit()
        
    cap = cv2.VideoCapture(0)
    
    # --- 1. Calibrate (Test) ---
    # We must iterate over the generator
    for frame in detector.calibrate(cap, calibration_time_sec=10):
        cv2.imshow("Calibrating...", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()
    
    if not detector.is_calibrated:
        print("Calibration failed. Exiting.")
        cap.release()
        exit()

    # --- 2. Detect (Test) ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result = detector.process_frame(frame)
        
        display_frame = frame.copy()
        
        if "error" in result:
            text = result["error"]
            color = (0, 0, 255)
        else:
            is_anomaly = result["is_anomaly"]
            text = "ANOMALY" if is_anomaly else "NORMAL"
            color = (0, 0, 255) if is_anomaly else (0, 255, 0)
            
            # --- Draw Heatmap ---
            heatmap = result["heatmap"]
            # Normalize for display
            if np.max(heatmap) > np.min(heatmap):
                # Scale heatmap to be more sensitive visually
                heatmap_norm = np.clip((heatmap - np.min(heatmap)) / (self.threshold - np.min(heatmap)), 0, 1)
            else:
                heatmap_norm = np.zeros_like(heatmap)
                
            heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)
            heatmap_resized = cv2.resize(heatmap_uint8, (frame.shape[1], frame.shape[0]))
            heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            
            # Overlay only if anomaly
            if is_anomaly:
                display_frame = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)
            # --- End Heatmap ---

        cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Live Anomaly Detection (DINOv2)", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

