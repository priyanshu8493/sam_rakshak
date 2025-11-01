import cv2
import time
import numpy as np
import os
from flask import Flask, Response, render_template, Blueprint, url_for as flask_url_for
from detector_core import AnomalyDetector # This is the new DINOv2 version
# data_handler is now used by detector_core, not directly by app.py

# --- 1. Initialize Flask App ---
app = Flask(__name__)

# --- 2. Initialize Detector Engine (UN-CALIBRATED) ---
print("[Server] Loading DINOv2 anomaly detector engine...")
# This will take a moment and may download the 340MB model
detector = AnomalyDetector() 
if not detector.model:
    print("[Server] ERROR: DINOv2 model not found or loaded. Detector will not work.")
print("[Server] Engine loaded. Detector is NOT calibrated.")

# --- 3. Blueprint & Production Prefix Setup ---
# This remains the same as your file
is_production = os.environ.get('FLASK_ENV') == 'production'
main = Blueprint('main', __name__)

def get_prefix():
    if is_production:
        return '/sam_rakshak'
    return ''

def custom_url_for(endpoint, **values):
    if endpoint.startswith('static'):
        return flask_url_for(endpoint, **values)
    if not endpoint.startswith('main.'):
        endpoint = f'main.{endpoint}'
    url = flask_url_for(endpoint, **values)
    if is_production:
        if not url.startswith('/'):
            url = '/' + url
        return f"{get_prefix()}{url}"
    return url

app.jinja_env.globals['url_for'] = custom_url_for

# --- 4. Define Video Generation Function (DINOv2 Version) ---
def gen_frames():
    """
    A generator function that yields processed video frames.
    Phase 1: Calibrate for 60 seconds (by building memory bank).
    Phase 2: Detect anomalies and draw heatmaps.
    """
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Video Thread] Error: Could not open camera.")
        # Create and yield a single error frame
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Camera Error: Could not access webcam", 
                    (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return # Stop the generator

    # --- PHASE 1: CALIBRATION (User-facing) ---
    if not detector.is_calibrated:
        print("[Video Thread] Starting 60-second DINOv2 calibration...")
        try:
            # Iterate over the calibrate generator
            for frame in detector.calibrate(cap, calibration_time_sec=60):
                # Encode and yield the "calibrating" frame
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"[Video Thread] Error during calibration: {e}")
            cap.release()
            return

        if not detector.is_calibrated:
            print("[Video Thread] Calibration FAILED.")
            cap.release()
            return
            
        print(f"[Video Thread] Calibration complete. Threshold: {detector.threshold:.4f}")

    # --- PHASE 2: DETECTION ---
    
    print("[Video Thread] Starting live DINOv2 detection...")
    while True:
        success, frame = cap.read()
        if not success:
            break

        result = detector.process_frame(frame)
        display_frame = frame.copy() # We'll draw on this copy
        
        if "error" in result:
            text = result["error"]
            color = (0, 0, 255)
            score_text = "Score: N/A"
        else:
            score = result["global_score"]
            is_anomaly = result["is_anomaly"]
            heatmap = result["heatmap"]
            score_text = f"Score: {score:.4f}"
            
            if is_anomaly:
                text = "ANOMALY DETECTED!"
                color = (0, 0, 255) # Red
                
                # --- Draw Heatmap Overlay ---
                if np.max(heatmap) > np.min(heatmap):
                    heatmap_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
                else:
                    heatmap_norm = np.zeros_like(heatmap)
                    
                heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)
                heatmap_resized = cv2.resize(heatmap_uint8, (frame.shape[1], frame.shape[0]))
                heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
                
                # Blend the heatmap
                display_frame = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)
                # --- End Heatmap ---
                
            else:
                text = "NORMAL"
                color = (0, 255, 0) # Green
        
        # Put text on the display frame
        cv2.putText(display_frame, text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display_frame, score_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Encode and yield the "detection" frame
        try:
            ret, buffer = cv2.imencode('.jpg', display_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"[Video Thread] Error encoding frame: {e}")

    print("[Video Thread] Releasing camera.")
    cap.release()


# --- 5. Define Routes ---
# (These routes are identical to your file)
@main.route('/')
def index():
    """Serves the main landing page."""
    return render_template('landing.html')

@main.route('/dashboard')
def dashboard():
    """Serves the dashboard page where the feed is shown."""
    return render_template('dashboard.html')

@main.route('/video_feed')
def video_feed():
    """The route that streams the video feed."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Register blueprint
app.register_blueprint(main, url_prefix=get_prefix())

# --- 6. Run the App ---
if __name__ == "__main__":
    if not detector.model:
        print("\n[Server] FATAL ERROR: DINOv2 model could not be loaded.")
        print("Please ensure you have an internet connection.")
    
    print("[Server] Starting Flask app...")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)

