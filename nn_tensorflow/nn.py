import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
import os

def load_images(path):
    images = []
    labels = []
    print("=============================")
    print("Loading images from: ", path)
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            for file in os.listdir(path + dir):
                full_path = os.path.join(path, dir, file)
                
                img = cv2.imread(full_path, 0)
                img = img[int(img.shape[0]/3):,:]
                img = cv2.Canny(img, 100, 200)
                img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)

                label = float(full_path.split('_')[1].replace('.jpg', ''))
                label = (label+1)*0.5
                
                images.append(img.flatten())
                labels.append(label)
                
    print("Done loading images:")
    print("Number of images: ", len(images))
    print("=============================")
    return (images, labels)


images, labels = load_images('data\\2\\')

images = np.asarray(images)
labels = np.asarray(labels)


model = keras.Sequential([
    keras.layers.Input(shape=(3200,)),
    keras.layers.Dense(2500, activation='sigmoid'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='mse')

model.summary()

model.fit(
    images, 
    labels, 
    epochs=5,
    batch_size=10,
    validation_split=0.1,
    shuffle=True
)

for i, img in enumerate(images):
    
    img = np.expand_dims(img, axis=0)
    print(f"Should be: {labels[i]}")
    print(f"Prediction: {model(img)}")
    img = img.reshape(40, 80)
    cv2.imshow('image', img)
    cv2.waitKey(0)