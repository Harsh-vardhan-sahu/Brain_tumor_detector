import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('BrainTumor10Epochs.h5')

# Load and preprocess the test image
image_path = 'C:\\Users\\91700\\Downloads\\IP_Project\\dataset\\yes\\yes.jpg'
image = cv2.imread(image_path)  # Read image
image = Image.fromarray(image, 'RGB')  # Convert to RGB (in case it's BGR)
image = image.resize((64, 64))  # Resize to match model input size
image = np.array(image)  # Convert to NumPy array

# Normalize the image (same as training process)
image = image / 255.0

# Expand dimensions to match model input shape (1, 64, 64, 3)
image = np.expand_dims(image, axis=0)

# Predict using the model
prediction = model.predict(image)

# Convert prediction to human-readable label
if prediction[0][0] > 0.5:  
    result = "Tumor Detected"
else:
    result = "No Tumor"

# Print result
print(f"Prediction: {result}")
