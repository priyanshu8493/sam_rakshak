from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf # Import full tensorflow for the loss function

def create_custom_loss(alpha=0.15, beta=0.85):
    """
    Creates a custom loss function combining MSE and (1 - SSIM).
    
    SSIM (Structural Similarity) is a value from -1 to 1. 1 means perfect similarity.
    (1 - SSIM) makes it a loss function (0 is perfect, 1 is bad).
    
    Args:
        alpha (float): Weight for MSE (pixel-level error).
        beta (float): Weight for SSIM (structural-level error).
    """
    def custom_loss(y_true, y_pred):
        # 1. Calculate Mean Squared Error (MSE)
        mse = K.mean(K.square(y_true - y_pred))
        
        # 2. Calculate Structural Similarity (SSIM)
        # tf.image.ssim returns a similarity score per image in the batch.
        # We must use max_val=1.0 since our pixels are normalized (0-1).
        ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
        
        # 3. Convert SSIM (a similarity) to a loss (a dissimilarity)
        # We average the SSIM score across the batch.
        ssim_loss = 1.0 - K.mean(ssim)
        
        # 4. Combine the two losses
        total_loss = (alpha * mse) + (beta * ssim_loss)
        
        return total_loss
    
    return custom_loss

def build_autoencoder(height, width, channels=1):
    """
    Builds the Convolutional Autoencoder (CAE) model.
    """
    K.clear_session()
    
    # --- The ENCODER ---
    inputs = Input(shape=(height, width, channels))
    
    # 128x128x1 -> 64x64x16
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    # 64x64x16 -> 32x32x32
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    # 32x32x32 -> 16x16x64
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    # --- NEW BOTTLENECK LAYER ---
    # 16x16x64 -> 8x8x128
    # This is the new, tighter bottleneck
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    # --- END NEW LAYER ---
    
    # --- The DECODER ---

    # --- NEW BOTTLENECK LAYER (MIRRORED) ---
    # 8x8x128 -> 16x16x128
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    # --- END NEW LAYER ---

    # 16x16x128 -> 32x32x64
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    
    # 32x32x64 -> 64x64x32
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    
    # 64x64x32 -> 128x128x16
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    
    # Final 'sigmoid' activation scales output between 0 and 1
    decoded = Conv2D(channels, (3, 3), activation='sigmoid', padding='same')(x)
    
    # --- Build and Compile ---
    autoencoder = Model(inputs, decoded)
    
    # Use our new, more powerful loss function
    ssim_mse_loss = create_custom_loss(alpha=0.15, beta=0.85)
    
    autoencoder.compile(optimizer='adam', loss=ssim_mse_loss)
    
    return autoencoder

if __name__ == "__main__":
    """
    A simple test to build the model and print its summary.
    """
    from data_handler import IMG_HEIGHT, IMG_WIDTH
    
    print("Building autoencoder...")
    model = build_autoencoder(IMG_HEIGHT, IMG_WIDTH, channels=1)
    model.summary()
    print("\nModel built successfully.")

