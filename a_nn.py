from typing import *

import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import utils, angles

class ACustomNN:
        def __init__(self, input_nodes :int, hidden_nodes :int, output_nodes :int, learning_rate :float, import_network=False):
            self.i_nodes = input_nodes
            self.h_nodes = hidden_nodes
            self.o_nodes = output_nodes

            self.learning_rate = learning_rate

            if import_network:
                with open('neuralnet.npy', 'rb') as f:
                    self.w_input_hidden = np.load(f)
                    self.w_hidden_output = np.load(f)
            else:
                self.w_input_hidden = np.random.rand(self.h_nodes, self.i_nodes) - 0.5
                self.w_hidden_output = np.random.rand(self.o_nodes, self.h_nodes) - 0.5

            self.activiation_function = lambda x : scipy.special.expit(x) # sigmoid
        pass
    
        def train(self, input_list, target_list):
            inputs = np.array(input_list, ndmin=2).T
            
            
            targets = np.array(target_list)


            h_inputs = np.dot(self.w_input_hidden, inputs)
            h_outputs = self.activiation_function(h_inputs)
            final_inputs = np.dot(self.w_hidden_output, h_outputs)
            final_outputs = self.activiation_function(final_inputs)

            # find errors
            output_errors = targets - final_outputs

            # errors into hidden layer
            hidden_errors = np.dot(self.w_hidden_output.T, output_errors)

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

def main():
    
    # the steering angles
    # targets :np.ndarray = np.load('labels.npy')
    # targets = (lambda x : x[0])(targets) # use only the 'steer' value
    targets = np.array(angles.ANGLES)
    print(targets.shape)
    
    # the images
    # inputs :np.ndarray = np.load('images.npy')
    images = utils.get_images("Testdaten/")
    inputs = np.array(images)
    
    print(inputs.shape)
    print(type(inputs[0]))
    
    
    input_nodes :int = inputs.shape[1] * inputs.shape[2] * inputs.shape[3] # all pixel values of an image
    hidden_nodes :int = 100
    output_nodes :int = 1
    learning_rate :int = 0.3
    epochs :int = 1
    
    network = ACustomNN(input_nodes, hidden_nodes, output_nodes, learning_rate)
    
    # start training
    for i in range(1, len(inputs)):
        print(f'--- Image number {i} ---')
        values :np.ndarray = inputs[i]
        network.train(values.flatten(), targets[i])
    pass

    print('\n!!!! training finished !!!')
    print("\nlet's test the network..")
    
    outputs = network.query(inputs[0].flatten())
    print('outputs: ', outputs)
    print('should be: ', targets[98])
    


if __name__ == '__main__':
    
    main()
    
    
    pass

