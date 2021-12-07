import NeuralNetwork as nn
import numpy as np
from matplotlib import pyplot as plt
# import tensorflow as tf
import sys
from image import Image

if __name__ == '__main__':
    image: Image = Image(False)
    if len(sys.argv) == 2:
        image: Image = Image(sys.argv[1])
    
    while True:
        image.set_speed_and_angle(0, 0)
        data_point = image.get_data_point()
        # prediction = nn.predict(data_point)
