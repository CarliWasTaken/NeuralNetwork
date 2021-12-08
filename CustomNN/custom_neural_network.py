from typing import *

import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import utils, angles
import random

# random.seed(1)
# np.random.seed(1)

class CustomNeuralNetwork:
    def __init__(self, input_nodes :int, hidden_nodes :int, output_nodes :int, learning_rate :float, import_network=False):
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes

        self.learning_rate = learning_rate

        if import_network:
            with open('network_data/neuralnet.npy', 'rb') as f:
                self.w_input_hidden = np.load(f)
                self.w_hidden_output = np.load(f)
        else:
            
            self.w_input_hidden = np.random.rand(self.h_nodes, self.i_nodes) - 0.5
            self.w_hidden_output = np.random.rand(self.o_nodes, self.h_nodes) - 0.5
            
        print('creating NN using the following parameters:')
        print('input_nodes: ', input_nodes)
        print('hidden_nodes: ', hidden_nodes)
        print('output_nodes: ', output_nodes)
        print('learning_rate: ', learning_rate)

        self.activiation_function = lambda x : scipy.special.expit(x) # sigmoid

    def train(self, input_list, target_list):
        inputs = np.array(input_list, ndmin=2).T
        # print('inputs shape:', inputs.shape)
        
        
        targets = np.array([target_list]).T
        # print(targets)
        # print('targets shape:', targets.shape)


        h_inputs = np.dot(self.w_input_hidden, inputs)
        h_outputs = self.activiation_function(h_inputs)
        final_inputs = np.dot(self.w_hidden_output, h_outputs)
        final_outputs = self.activiation_function(final_inputs)
        
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
        hidden_errors = np.dot(self.w_hidden_output.T, output_errors)
        # print('hidden_errors shape:', hidden_errors.shape)

        # weight adjustments
        # alpha * output_errors * final_outputs * (1 - final_outputs) * transpose(hidden_outputs)
        self.w_hidden_output += self.learning_rate * np.dot((output_errors * final_outputs *  (1 - final_outputs)),  np.transpose(h_outputs))
        self.w_input_hidden += self.learning_rate * np.dot((hidden_errors * h_outputs * (1 - h_outputs)),  np.transpose(inputs))

        pass
    
    # input -> ann -> output
    def query(self, input_list):
        # aufdrösseln der input_list in etwas brauchbares
        inputs = np.array(input_list, ndmin=2).T
        

        # X(h) = I * W(i-h)
        h_inputs = np.dot(self.w_input_hidden, inputs)

        # O(h) = sigmoid(X(h)) 
        h_outputs = self.activiation_function(h_inputs)

        # X(o) = O(h) * W(h-o)
        final_inputs = np.dot(self.w_hidden_output, h_outputs)

        # O = sigmoid(X(o))
        final_outputs = self.activiation_function(final_inputs)

        return final_outputs
    
    def debug_net(self):
        print('w_input_hidden')
        print(self.w_input_hidden)
        print('w_hidden_output')
        print(self.w_hidden_output)

        pass
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            np.save(f, self.w_input_hidden)
            np.save(f, self.w_hidden_output)
    
    @classmethod            
    def import_neural_net(cls, filename, learning_rate=0.2):
        with open(filename, 'rb') as f:
            print('importing network...')
            
            w_input_hidden = np.load(f)
            w_hidden_output = np.load(f)
            
            print('weights_input_hidden: ', w_input_hidden.shape)
            print('weights_hidden_output: ', w_hidden_output.shape)
            
            input_nodes = w_input_hidden.shape[1]
            hidden_nodes = w_hidden_output.shape[1]
            output_nodes = w_hidden_output.shape[0]
            
            nn = CustomNeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate, import_network=True)
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
    
   
    target_list = np.array(angles.ANGLES)
    print(target_list.shape)
    
    # the images
    # inputs :np.ndarray = np.load('images.npy')
    directories = ('Testdaten/left', 'Testdaten/right', 'Testdaten/Straight')
    inputs = []
    
    for d in directories:
        images = utils.get_images(d)
        inputs.extend(images)
    
    inputs = np.array(inputs)
    
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
        values :np.ndarray = (inputs[i] / 255.0 * 0.99)
        targets = np.zeros(output_nodes) + 0.01
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
    
    
    
    main()
    
    
    pass

