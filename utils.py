from PIL import Image
import PIL
import os
import glob

import numpy as np
import cv2



picture = Image.open("Compressed_image-1-compressed.jpg")
file_name = 'image-1-compressed.jpg'
dim = picture.size
print(f"This is the current width and height of the image: {dim}")

def get_images(path):
    files = os.listdir(path)
    return [cv2.imread(os.path.join(path, file.title())) for file in files if file.endswith(('jpg', 'png', 'Jpg'))]

def compress_images(directory=False, quality=30):

    # 2. Extract all of the .png and .jpeg files:
    files = os.listdir("RawTestdaten/")

    # 3. Extract all of the images:
    images = [file for file in files if file.endswith(('jpg', 'png', 'Jpg'))]

    # 4. Loop over every image:
    for image in images:
        path = os.path.join("RawTestdaten/", image.title())
        print(path)
        img = cv2.imread(path, 1)
        img_half = cv2.resize(img, (0, 0), fx=0.02, fy=0.02)
        
        # cv2.imshow('Half Image', img_half)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        cv2.imwrite('Testdaten/' + image.title(), img_half)
        # break
        
# compress_images()

