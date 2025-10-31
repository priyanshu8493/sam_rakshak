import cv2
import numpy as np

# --- Constants ---
# Using a smaller, square image is much faster for the model
IMG_WIDTH = 128
IMG_HEIGHT = 128
# --- End Constants ---

def preprocess_frame(frame):
    """
    Converts a single video frame into a model-ready format.
    
    1. Converts to Grayscale
    2. Resizes to (IMG_HEIGHT, IMG_WIDTH)
    3. Normalizes pixel values to be between 0.0 and 1.0
    4. Reshapes for Keras: (1, height, width, 1)
    
    Args:
        frame (numpy.ndarray): The raw video frame from OpenCV.
        
    Returns:
        numpy.ndarray: The processed frame, ready for the model.
    """
    try:
        # 1. Convert to Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Resize to our standard size
        resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
        
        # 3. Normalize pixel values
        normalized = resized.astype("float32") / 255.0
        
        # 4. Reshape for Keras (batch_size, height, width, channels)
        return np.reshape(normalized, (1, IMG_HEIGHT, IMG_WIDTH, 1))
    
    except cv2.error as e:
        print(f"[DataHandler] OpenCV Error: {e}")
        return None
    except Exception as e:
        print(f"[DataHandler] Error processing frame: {e}")
        return None

if __name__ == "__main__":
    """
    A simple test script to show the preprocessor in action.
    This opens your webcam and displays the original vs. processed feed.
    """
    
    cap = cv2.VideoCapture(0) # 0 is your default webcam
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
        
    print("Opening webcam... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            break
            
        processed_input = preprocess_frame(frame)
        
        if processed_input is not None:
            # To display the preprocessed image, we 'un-normalize' and 'un-reshape' it
            display_frame = (processed_input.squeeze() * 255).astype("uint8")
            cv2.imshow("Processed (Grayscale, 128x128)", display_frame)
        
        cv2.imshow("Original Feed", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

