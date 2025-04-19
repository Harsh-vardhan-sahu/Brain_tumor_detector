import cv2
import os
from PIL import Image
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split  
from keras.utils import normalize, to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

# Dataset path
img = 'dataset/'
dataset = []
label = []

# Get images
no_tumor = os.listdir(img + 'no/')
yes_tumor = os.listdir(img + 'yes/')

INPUT_SIZE = 64

# Loop for "no tumor" images
for image_name in no_tumor:
    if image_name.endswith('.jpg'):  
        image = cv2.imread(img + 'no/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

# Loop for "yes tumor" images
for image_name in yes_tumor:
    if image_name.endswith('.jpg'):
        image = cv2.imread(img + 'yes/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

# Convert dataset and label to NumPy arrays
dataset = np.array(dataset)
label = np.array(label)

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

# Normalize dataset (scaling pixel values)
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

# ✅ FIX: Convert labels to categorical (one-hot encoding)
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Model Building
model = Sequential()

# First Conv Layer
model.add(Conv2D(32, (3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Second Conv Layer
model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Third Conv Layer
model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Fully Connected Layer
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

# Compile the Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the Model
model.fit(x_train, y_train, batch_size=1, verbose=1, epochs=35, validation_data=(x_test, y_test), shuffle=True)

# Save the Model
model.save('BrainTumor40Epochs_Categorical.h5')
