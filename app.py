import cv2
import time
from flask import Flask, Response, render_template
from detector_core import AnomalyDetector
from data_handler import preprocess_frame # We need this for calibration
import numpy as np # <-- 1. IMPORT NUMPY

# --- 1. Initialize Flask App ---
app = Flask(__name__)

# --- 2. Initialize Detector Engine (UN-CALIBRATED) ---
print("[Server] Loading anomaly detector engine...")
detector = AnomalyDetector()
if not detector.model:
    print("[Server] ERROR: Model file not found. Detector will not work.")
print("[Server] Engine loaded. Detector is NOT calibrated.")


# --- 3. Define Video Generation Function (with 2 phases) ---
def gen_frames():
    """
    A generator function that yields processed video frames.
    Phase 1: Calibrate for 60 seconds.
    Phase 2: Detect anomalies.
    """
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Video Thread] Error: Could not open camera.")
        return

    # --- PHASE 1: CALIBRATION (User-facing) ---
    
    # Check if detector needs calibration
    if not detector.is_calibrated:
        print("[Video Thread] Starting 60-second calibration...")
        
        calibration_time_sec = 60
        start_time = time.time()
        normal_errors = []

        while time.time() - start_time < calibration_time_sec:
            success, frame = cap.read()
            if not success:
                break
            
            # Preprocess the frame for the model
            processed = preprocess_frame(frame)
            if processed is not None:
                reconstruction = detector.model.predict(processed, verbose=0)
                
                # --- THIS IS THE FIX ---
                # Calculate MSE directly, instead of calling a non-existent method
                mse = np.mean(np.square(processed - reconstruction))
                # --- END FIX ---
                
                normal_errors.append(mse)

            # Draw calibration feedback on the frame
            remaining = int(calibration_time_sec - (time.time() - start_time))
            text = f"CALIBRATING... PLEASE WAIT ({remaining}s)"
            cv2.putText(frame, text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # Yellow
            cv2.putText(frame, "Please show 'normal' activity", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Yield the "calibrating" frame
            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                print(f"[Video Thread] Error encoding frame: {e}")
        
        # Now, finish calibration
        if normal_errors:
            # --- THIS IS THE FIX ---
            # Set the threshold directly on the detector object
            detector.threshold = np.mean(normal_errors) + (np.std(normal_errors) * 3)
            detector.is_calibrated = True
            # --- END FIX ---
            print(f"[Video Thread] Calibration complete. Threshold: {detector.threshold:.4f}")
        else:
            print("[Video Thread] Calibration FAILED. No frames read.")
            # We'll just set a default to prevent a crash, though detection won't be good
            detector.threshold = 0.01 
            detector.is_calibrated = True

    # --- PHASE 2: DETECTION ---
    
    print("[Video Thread] Starting live detection...")
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Get anomaly score and draw on the frame
        result = detector.process_frame(frame)
        
        if "error" in result:
            text = result["error"]
            color = (0, 0, 255)
            score = 0.0
        else:
            score = result["mse_score"]
            is_anomaly = result["is_anomaly"]
            
            if is_anomaly:
                text = "ANOMALY DETECTED!"
                color = (0, 0, 255) # Red
            else:
                text = "NORMAL"
                color = (0, 255, 0) # Green
        
        # Put text on the frame
        cv2.putText(frame, text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Score: {score:.4f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Encode and yield the "detection" frame
        try:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"[Video Thread] Error encoding frame: {e}")

    print("[Video Thread] Releasing camera.")
    cap.release()


# --- 4. Define Routes ---

@app.route('/')
def index():
    """Serves the main landing page."""
    return render_template('landing.html')

@app.route('/dashboard')
def dashboard():
    """Serves the dashboard page where the feed is shown."""
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    """The route that streams the video feed."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- 5. Run the App ---
if __name__ == "__main__":
    if not detector.model:
        print("\n[Server] FATAL ERROR: 'anomaly_detector.h5' not found.")
        print("Please run train.py first to create the model file.")
    
    print("[Server] Starting Flask app...")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)

