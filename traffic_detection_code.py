# Fundamental classes
import numpy as np
import pandas as pd
import tensorflow as tf
import os

# Image related
import cv2
from PIL import Image

# For plotting
import matplotlib.pyplot as plt

# For the model and its training
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

# Setting variables
data = []
labels = []
classes = 43
cur_path = os.getcwd()

# Retrieving images and labels
print("Loading training images...")
for i in range(classes):
    path = os.path.join('german_traffic_sign_dataset', 'Train', str(i))
    images = os.listdir(path)

    for a in images:
        try:
            image = Image.open(path + '/' + a)
            image = image.resize((30, 30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print(f"Error loading image: {a}")

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# One-hot encode labels
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

# Building the CNN model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
print("Training model...")
with tf.device('/GPU:0'):
    epochs = 5
    history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))

# Testing accuracy on separate test dataset
print("Evaluating on test data...")
test_df = pd.read_csv('german_traffic_sign_dataset/Test.csv')
labels = test_df["ClassId"].values
imgs = test_df["Path"].values

test_data = []
with tf.device('/GPU:0'):
    for img in imgs:
        image = Image.open('german_traffic_sign_dataset' + img)
        image = image.resize((30, 30))
        test_data.append(np.array(image))

X_test = np.array(test_data)

with tf.device('/GPU:0'):
    pred = np.argmax(model.predict(X_test), axis=-1)

# Output accuracy
print("Test accuracy:", accuracy_score(labels, pred))

# Save the model
model.save('traffic_classifier.h5')
print("Model saved as 'traffic_classifier.h5'")
