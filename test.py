import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import cv2
from keras.models import load_model

# Load the pre-trained model
model = load_model('BrainTumor10Epochs.h5')

# Prediction function
def predict_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to load image at {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = image.resize((64, 64))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    
    prediction = model.predict(image)
    return "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor"

# Browse and display the image
def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((250, 250))
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk
        result_label.config(text="Predicting...", fg="blue")
        root.update()

        result = predict_image(file_path)
        color = "red" if result == "Tumor Detected" else "green"
        result_label.config(text=f"Prediction: {result}", fg=color)

# GUI setup
root = tk.Tk()
root.title("Brain Tumor Detection")
root.geometry("400x450")
root.resizable(False, False)

title_label = Label(root, text="Brain Tumor Detector", font=("Arial", 18, "bold"))
title_label.pack(pady=10)

panel = Label(root)
panel.pack(pady=10)

browse_button = Button(root, text="Browse Image", command=browse_image, font=("Arial", 12))
browse_button.pack(pady=10)

result_label = Label(root, text="Upload an image to start prediction", font=("Arial", 14))
result_label.pack(pady=20)

# Run the GUI loop
root.mainloop()
