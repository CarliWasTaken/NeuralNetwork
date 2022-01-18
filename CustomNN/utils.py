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
    # images = []
    # for file in files:
    #     if file.endswith(('jpg', 'png', 'Jpg')):
    #         image = cv2.imread(os.path.join(path, file.title()),0)
    #         images.append(image)
            
    return [cv2.imread(os.path.join(path, file.title()),0) for file in files if file.endswith(('jpg', 'png', 'Jpg'))] # 0 for gray
    # return images

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
        cv2.cvtColor(img, cv2.COLOR_)
        
        
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
        


def load_data(path):
    # iterate over all images in data folder
    files = os.listdir(path)

    # 1. Extract all of the images:
    images = [file for file in files if file.endswith(('jpg', 'png', 'Jpg'))]
    
    training_images = []
    training_labels = []

    # 2. Loop over every image:
    for image in images:
        img = cv2.imread(path+image, 0)
        label = image.split('_')[2].replace('.jpg', '')
        
        # remove upper 1/3 of image
        img =  img[img.shape[0]//3:, :]
        
        temp_img = cv2.Canny(img, 50, 70)
        # temp_img = cv2.resize(temp_img, (0, 0), fx=10, fy=10)
        # cv2.imshow('Half Image', temp_img)
        # cv2.imshow('Full Image', img)
        # cv2.moveWindow('Full Image', 1700, 800)
        # cv2.moveWindow('Half Image', 1700, 1000)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # raise Exception('remove this part')
        img = temp_img
        

        training_images.append(img)
        training_labels.append(float(label))
        
    return (training_images, training_labels)
        
def rename(path):
    counter = 0
    for file in os.listdir(path):
        if file.endswith('.jpg'):
            label = file.split('_')[1].replace('.jpg', '')
            os.rename(os.path.join(path, file), os.path.join(path, f'image_{counter}_{label}.jpg'))
            counter += 1

if __name__ == '__main__':
    '''down_scale("RawTestData/left/", "TestData/left/")
    down_scale("RawTestData/right/", "TestData/right/")
    down_scale("RawTestData/Straight/", "TestData/Straight/")'''
    
    
    # data = format_training_data('Data/data/training_data/')
    # print(data)
    # rename('Data/data/training_data/')
