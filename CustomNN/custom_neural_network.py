from typing import *

import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import utils, angles
import random
import cupy as cp
from time import time

# random.seed(1)
# cp.random.seed(1)

class CustomNeuralNetwork:
    def __init__(self, input_nodes :int, hidden_nodes :int, output_nodes :int, learning_rate :float, weights :Tuple[cp.ndarray, cp.ndarray]=None):
        '''Creates a wonderful Neural Network (NN)
        
        Parameters
        
            input_nodes :int
                The number of input nodes
            hidden_nodes :int
                The number of hidden nodes
            output_nodes :int
                The number of output nodes
            learning_rate :float
                The learning rate of the network
            weights :Tuple[cp.ndarray, cp.ndarray], optional
                The weights of the network. If not given, the weights are initialized randomly
                Structure: (weights_input_hidden, weights_hidden_output)
        
        '''
  
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes

        self.learning_rate = learning_rate

        # initialize weights (or use the given ones)
        if weights:
            self.w_input_hidden = weights[0]
            self.w_hidden_output = weights[1]
        else:
            self.w_input_hidden = cp.random.rand(self.h_nodes, self.i_nodes) - 0.5
            self.w_hidden_output = cp.random.rand(self.o_nodes, self.h_nodes) - 0.5
            
        print('creating NN using the following parameters:')
        print('input_nodes: ', input_nodes)
        print('hidden_nodes: ', hidden_nodes)
        print('output_nodes: ', output_nodes)
        print('learning_rate: ', learning_rate)
        
        expit = cp.ElementwiseKernel('float64 x', 'float64 y', 'y = 1 / (1 + exp(-x))', 'expit')

        self.activiation_function = lambda x : scipy.special.expit(x) # sigmoid

    def train(self, input_list, target_list):
        '''Trains the network with the given input and target
        
        Parameters
        
            input_list :cp.ndarray
                The list of input_nodes (= one image)
            target_list :cp.ndarray
                The list of output_nodes (= percentages for: left, right, straight)
        
        '''
        
        inputs = cp.array(input_list, ndmin=2).T
        # print('inputs shape:', inputs.shape)
        
        
        targets = cp.array([target_list]).T
        # print(targets)
        # print('targets shape:', targets.shape)


        h_inputs = cp.dot(self.w_input_hidden, inputs)
        h_outputs = cp.asarray(self.activiation_function(h_inputs.get()))
        final_inputs = cp.dot(self.w_hidden_output, h_outputs)
        final_outputs = cp.asarray(self.activiation_function(final_inputs.get()))
        
        # print('w_input_hidden shape:', self.w_input_hidden.shape)
        # print('w_hidden_output shape:', self.w_hidden_output.shape)
        
        # print('h_inputs shape:', h_inputs.shape)
        # print('h_outputs shape:', h_outputs.shape)
        # print('final_inputs shape:', final_inputs.shape)
        # print('final_outputs shape:', final_outputs.shape)

        # find errors
        output_errors = targets - final_outputs
        # print('output_errors shape:', output_errors.shape)

        # errors into hidden layer
        hidden_errors = cp.dot(self.w_hidden_output.T, output_errors)
        # print('hidden_errors shape:', hidden_errors.shape)

        # weight adjustments
        # alpha * output_errors * final_outputs * (1 - final_outputs) * transpose(hidden_outputs)
        self.w_hidden_output += self.learning_rate * cp.dot((output_errors * final_outputs *  (1 - final_outputs)),  cp.transpose(h_outputs))
        self.w_input_hidden += self.learning_rate * cp.dot((hidden_errors * h_outputs * (1 - h_outputs)),  cp.transpose(inputs))

        pass
     
    def query(self, input_list :cp.ndarray) -> cp.ndarray:
        '''Uses the network to predict the output of a given input
        
        Parameters
        
            input_list :cp.ndarray
                the list of input_nodes (= one image)
        
        Returns
        
            output_nodes :cp.ndarray
                the list of output_nodes (= percentages for: left, right, straight)
        '''
        # input -> ann -> output
        # t1 = time()
        
        # aufdrösseln der input_list in etwas brauchbares
        inputs = cp.array(input_list, ndmin=2).T
        

        # X(h) = I * W(i-h)
        h_inputs = cp.dot(self.w_input_hidden, inputs)
        
        # O(h) = sigmoid(X(h)) 
        h_outputs = cp.asarray(self.activiation_function(h_inputs.get()))
        
        # X(o) = O(h) * W(h-o)
        final_inputs = cp.dot(self.w_hidden_output, h_outputs)
        
        # O = sigmoid(X(o))
        final_outputs = cp.asarray(self.activiation_function(final_inputs.get()))
        
        # print('query time:', time() - t1)

        return final_outputs
    
    def debug_net(self):
        '''prints all the weights of the network'''
        
        print('w_input_hidden')
        print(self.w_input_hidden)
        print('w_hidden_output')
        print(self.w_hidden_output)

        pass
    
    def save(self, filename :str):
        '''Saves all the weights of the network to a file 
        
        using the following structure:
        
            - weights_input_hidden
            - weights_hidden_output
        
        Note: The learning rate is not saved
        
        Parameters
        
            filename :str
                The name of the file (.npy)
        
        '''
        with open(filename, 'wb') as f:
            cp.save(f, self.w_input_hidden)
            cp.save(f, self.w_hidden_output)
    
    @classmethod            
    def import_neural_net(cls, filename :str, learning_rate :float = 0.2):
        '''Imports all the weights for the NN from a given file
        Note: The learning rate is not imported.
        
        Parameters
        
            filename: str 
                The Name of the file to import the weights from (.npy)
                File-Structure: weights_input_hidden, weights_hidden_output
            learning_rate: float, optional (default=0.2)
                The learning rate of the NN
        '''
        
        
        with open(filename, 'rb') as f:
            print('importing network...')
            
            w_input_hidden = cp.load(f)
            w_hidden_output = cp.load(f)
            
            print('weights_input_hidden: ', w_input_hidden.shape)
            print('weights_hidden_output: ', w_hidden_output.shape)
            
            input_nodes = w_input_hidden.shape[1]
            hidden_nodes = w_hidden_output.shape[1]
            output_nodes = w_hidden_output.shape[0]
            
            nn = CustomNeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate, weights=(w_input_hidden, w_hidden_output))
            return nn



def get_training_and_test_indices(number_of_records, number_of_types=3, number_of_test_records_per_type=5):
    start = number_of_records // number_of_types
    training_indices = list(range(number_of_records))
    test_indices = [training_indices[start*i:start*i+number_of_test_records_per_type] for i in range(number_of_types)]
    test_indices = set([item for sublist in test_indices for item in sublist])
    training_indices = list(set(training_indices) - test_indices)   
    random.shuffle(training_indices)
    return training_indices, list(test_indices)
        

def main():
    
   
    target_list = cp.array(angles.ANGLES)
    print(target_list.shape)
    
    # the images
    # inputs :cp.ndarray = cp.load('images.npy')
    directories = ('Testdaten/left', 'Testdaten/right', 'Testdaten/Straight')
    inputs = []
    
    for d in directories:
        images = utils.get_images(d)
        inputs.extend(images)
    
    inputs = cp.array(inputs)
    
    training_indices, test_indices = get_training_and_test_indices(len(inputs))
    
    
    print(inputs.shape)
    
    # shape of inputs should look like: (numberofimages, y, x, channels)
    # shape of targets should look like: (numberofvalues,)
    # where numberofvalues == numberofimages
    
    # init network
    input_nodes :int = inputs.shape[1] * inputs.shape[2]# * inputs.shape[3] # all pixels values of an image
    hidden_nodes :int = 3000
    output_nodes :int = 3
    learning_rate :int = 0.2
    
    network = CustomNeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    
    # train the network
    x = 0
    print('training started!\n')
    for i in training_indices:
        x += 1
        print(f'--- Image number {x, i} ---', end='\r')
        values :cp.ndarray = (inputs[i] / 255.0 * 0.99)
        targets = cp.zeros(output_nodes) + 0.01
        targets[int(target_list[i])] = 0.99
        network.train(values.flatten(), targets)
    pass

    print('\n\n!!!! training finished !!!')
    
    print('saving network...')
    network.save('network_data/neuralnet.npy')
    print('saved network successfully!')
    
    print("\nlet's test the network..")
    
    print('test_indices: ', test_indices)
    
    for i in test_indices:
        outputs = network.query(inputs[i].flatten())
        print(f'outputs [{i}]: \n', outputs)
        print('should be index: ', target_list[i], '\n')
        
    


if __name__ == '__main__':
    
    # main()
    
    
    pass

