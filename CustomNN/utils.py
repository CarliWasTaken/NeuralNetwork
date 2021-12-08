from PIL import Image
import PIL
import os
import glob

import numpy as np
import cv2

# picture = Image.open("Compressed_image-1-compressed.jpg")
# file_name = 'image-1-compressed.jpg'
# dim = picture.size
# print(f"This is the current width and height of the image: {dim}")

def get_images(path):
    files = os.listdir(path)
    return [cv2.imread(os.path.join(path, file.title()),0) for file in files if file.endswith(('jpg', 'png', 'Jpg'))] # 0 for gray

def down_scale(path_in, path_out):

    # 2. Extract all of the .png and .jpeg files:
    files = os.listdir(path_in)

    # 3. Extract all of the images:
    images = [file for file in files if file.endswith(('jpg', 'png', 'Jpg'))]

    # 4. Loop over every image:
    for image in images:
        path = os.path.join(path_in, image.title())
        print(path)
        img = cv2.imread(path, 1)
        # img = cv2.GaussianBlur(img,(81,81),0) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        
        # lower bound and upper bound 
        lower_bound = np.array([0, 0, 0])   
        upper_bound = np.array([140, 140, 140])
        img = cv2.inRange(img, lower_bound, upper_bound)

        
        # img = cv2.Canny(img, 10, 20)
        img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
        # print(img.shape)        
        
        # cv2.imshow('Half Image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # break
        cv2.imwrite(path_out + image.title(), img)
        

if __name__ == '__main__':
    down_scale("RawTestData/left/", "TestData/left/")
    down_scale("RawTestData/right/", "TestData/right/")
    down_scale("RawTestData/Straight/", "TestData/Straight/")

