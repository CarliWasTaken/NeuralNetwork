import cam
import NeuralNetwork as nn
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

if __name__ == '__main__':
    while True:
        data_point = cam.get_data_point()
        prediction = nn.predict(data_point)
