import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
import os

def load_images(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        label = (float(filename.split('_')[2].replace('.jpg', ''))+1)*0.5
        labels.append(label)
        img = cv2.imread(os.path.join(folder,filename), 0)
        if img is not None:
            images.append(img)
    return (images, labels)


images, labels = load_images('data')

#test_images = np.asarray(images[-int(len(images)*0.1):])
#test_labels = np.asarray(labels[-int(len(labels)*0.1):])

train_images = np.asarray(images)
train_labels = np.asarray(labels)


model = keras.Sequential([
    keras.layers.Input(shape=(48, 64)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='mse')

model.fit(
    train_images, 
    train_labels, 
    epochs=10,
    batch_size=1,
    validation_split=0.1,
    shuffle=True
)

model.save('model.h5')