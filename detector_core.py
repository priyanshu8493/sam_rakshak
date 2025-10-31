import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from data_handler import preprocess_frame, IMG_HEIGHT, IMG_WIDTH
from model import create_custom_loss 
class AnomalyDetector:
    """
    A backend-ready class to handle anomaly detection.
    
    This class loads the model, calibrates a threshold, and 
    provides a single method to check a frame for anomalies.
    """
    
    def __init__(self, model_path="anomaly_detector.h5"):
        print(f"[DetectorCore] Loading model from {model_path}...")
        try:
            # Tell load_model how to find our custom loss
            ssim_mse_loss = create_custom_loss()
            self.model = load_model(
                model_path, 
                custom_objects={'custom_loss': ssim_mse_loss}
            )
            
            self.threshold = None
            self.is_calibrated = False
            print("[DetectorCore] Model loaded.")
        except IOError:
            print(f"[DetectorCore] Error: Model file not found at {model_path}")
            print("[DetectorCore] Please run train.py first to create the model.")
            self.model = None
            self.is_calibrated = False

    def calibrate(self, video_source=0, calibration_time_sec=10):
        """
        Calibrates the anomaly threshold by observing "normal" video.
        
        Args:
            video_source: The webcam index (e.g., 0) or video file path.
            calibration_time_sec (int): How long to run calibration.
        """
        if not self.model:
            print("[DetectorCore] Error: Model is not loaded. Cannot calibrate.")
            return

        print(f"[DetectorCore] Starting calibration for {calibration_time_sec}s...")
        print("[DetectorCore] Please show ONLY 'normal' activity.")
        
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return

        try:
            # Give webcam time to warm up
            cv2.waitKey(2000)
            
            normal_errors = []
            start_time = time.time()
            
            while time.time() - start_time < calibration_time_sec:
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed = preprocess_frame(frame)
                if processed is None:
                    continue
                    
                reconstruction = self.model.predict(processed, verbose=0)
                mse = np.mean(np.square(processed - reconstruction))
                normal_errors.append(mse)
                
                # Optional: Show a calibration window
                cv2.putText(frame, "CALIBRATING...", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Calibrating...", frame)
                cv2.waitKey(1)

            if not normal_errors:
                print("[DetectorCore] Error: No frames captured during calibration.")
                return

            # Set threshold to be mean + 3 standard deviations
            self.threshold = np.mean(normal_errors) + (np.std(normal_errors) * 3)
            self.is_calibrated = True
            print(f"[DetectorCore] Calibration complete. Anomaly threshold set to: {self.threshold}")
        
        finally:
            # Ensure resources are released
            cap.release()
            cv2.destroyAllWindows()
            
        return self.threshold

    def process_frame(self, frame):
        """
        Processes a single frame and returns its anomaly status.
        
        Args:
            frame (numpy.ndarray): The raw video frame from OpenCV.
            
        Returns:
            dict: A dictionary with 'mse_score', 'is_anomaly', and 'threshold'.
        """
        if not self.is_calibrated:
            return {
                "error": "Detector not calibrated. Please run calibrate() first."
            }

        processed = preprocess_frame(frame)
        if processed is None:
            return {"error": "Invalid frame."}

        reconstruction = self.model.predict(processed, verbose=0)
        mse = np.mean(np.square(processed - reconstruction))
        
        is_anomaly = mse > self.threshold
        
        return {
            "mse_score": mse,
            "is_anomaly": is_anomaly,
            "threshold": self.threshold
        }

if __name__ == "__main__":
    """
    A simple test script to show how to use the AnomalyDetector class.
    
    1. Calibrates the detector.
    2. Runs a live feed to show detections.
    """
    
    detector = AnomalyDetector()
    
    if not detector.model:
        print("Exiting. Model file 'anomaly_detector.h5' not found.")
        print("Please run train.py first.")
        exit()
    
    # --- 1. Calibrate ---
    # This must be run once when your backend starts.
    detector.calibrate(video_source=0, calibration_time_sec=10)
    
    if not detector.is_calibrated:
        print("Exiting. Calibration failed.")
        exit()
        
    print("\n--- Starting Live Detection ---")
    print("Press 'q' to quit.")
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # --- 2. Process Frame ---
        # This is the function your backend API will call
        result = detector.process_frame(frame)
        
        if "error" in result:
            print(result["error"])
            continue

        mse = result["mse_score"]
        is_anomaly = result["is_anomaly"]
        
        if is_anomaly:
            cv2.putText(frame, "ANOMALY DETECTED!", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "NORMAL", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        cv2.putText(frame, f"Score: {mse:.4f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Live Anomaly Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()



