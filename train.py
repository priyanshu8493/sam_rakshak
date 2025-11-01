import cv2
import numpy as np
import time
from data_handler import preprocess_frame, IMG_HEIGHT, IMG_WIDTH
from model import build_autoencoder

# --- Constants ---
TRAINING_TIME_SEC = 160 # How long to record "normal" video.
FRAMES_TO_SKIP = 10    # Only save 1 out of every 5 frames to get diverse data.
MODEL_EPOCHS = 300   # How many times to loop over the data. 30-40 is good.
MODEL_BATCH_SIZE = 32  # How many frames to process at once.
# --- End Constants ---

def collect_training_data(video_source=0):
    """
    Opens the webcam and records "normal" frames for training.
    """
    print(f"[Trainer] Starting data collection for {TRAINING_TIME_SEC} seconds...")
    print("[Trainer] Please show ONLY 'normal' activity to the camera.")
    print("[Trainer] Starting in 5...")
    time.sleep(5)
    print("[Trainer] GO! Recording...")

    training_data = []
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return None
        
    start_time = time.time()
    frame_count = 0

    try:
        while time.time() - start_time < TRAINING_TIME_SEC:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            cv2.imshow("Training Data Collection (Press 'q' to stop early)", frame)
            
            # Only save 1 in every N frames
            if frame_count % FRAMES_TO_SKIP == 0:
                processed = preprocess_frame(frame)
                if processed is not None:
                    training_data.append(processed)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[Trainer] Data collection stopped early by user.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not training_data:
        print("[Trainer] Error: No frames collected. Check webcam.")
        return None

    print(f"[Trainer] Data collection complete. Collected {len(training_data)} frames.")
    
    # Stack all frames into a single big numpy array
    return np.vstack(training_data)

def train_model(train_x):
    """
    Trains the autoencoder model on the collected frames.
    """
    if train_x is None:
        return None

    print("[Trainer] Building model...")
    autoencoder = build_autoencoder(IMG_HEIGHT, IMG_WIDTH, channels=1)

    print("[Trainer] Training model... This might take a few minutes.")
    
    # We train the autoencoder to reconstruct itself.
    # So, x (input) and y (target) are the SAME data.
    autoencoder.fit(
        train_x,
        train_x, 
        epochs=MODEL_EPOCHS,
        batch_size=MODEL_BATCH_SIZE,
        shuffle=True,
        validation_split=0.1 # Use 10% of data for validation
    )

    return autoencoder

if __name__ == "__main__":
    # 1. Collect Data
    training_data = collect_training_data(video_source=0)
    
    if training_data is not None:
        # 2. Train Model
        model = train_model(training_data)
        
        if model is not None:
            # 3. Save the Model
            model.save("anomaly_detector.h5")
            print("\n[Trainer] Model training complete!")
            print("[Trainer] Model saved as 'anomaly_detector.h5'!")

