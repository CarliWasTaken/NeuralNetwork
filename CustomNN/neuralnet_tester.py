from typing import *

import numpy as np
from numpy.core.defchararray import count
import scipy.special
import matplotlib.pyplot as plt
import utils, angles
import random
import os
import cv2
import cupy as cp
import time



from custom_neural_network import CustomNeuralNetwork

# random.seed(1)
# np.random.seed(1)

class NeuralnetTester:
    def __init__(self, path_to_data :str = None, import_weight_path :str = None, input_nodes :int = 1200, hidden_nodes :int = 3000, output_nodes :int = 3, learning_rate :float = 0.2, image_scale :float = 0.4):
        
        '''The wonderful constructor of the NeuralnetTester class
        
        Parameters
        
            import_weight_path :str, optional (default=None)
                The path to the file containing the weights of the network
            input_nodes :int, optional (default=1200)
                The number of input nodes of the network
            hidden_nodes :int, optional (default=3000)
                The number of hidden nodes of the network
            output_nodes :int, optional (default=3)
                The number of output nodes of the network
            learning_rate :float, optional (default=0.2)
                The learning rate of the network
        
        '''
        
        self.image_scale = image_scale
        

        
        # load data (inputs + targets)
        if path_to_data:
            self.load_data(path_to_data, self.image_scale)
        
        # load data from directories
        # if paths_to_input_directories:
        #     print('\nloading input data from directories...')
        #     _inputs = []
        #     for a_dir in paths_to_input_directories:
        #         print(f'directory {a_dir}')
        #         images = utils.get_images(a_dir)
        #         _inputs.extend(images)
        #         print(f'found {len(images)} images')
        #     self.inputs = _inputs
              
        # convert to numpy array    
        self.inputs = np.array(self.inputs)
        self.original_targets = np.array(self.targets)
        self.targets = np.array(self.targets) / 2.0 + 0.5 # normalize targets (eliminate negative values)
        
        # print(f'targets {self.targets[200:205]}')
        # print(f'otargets {self.original_targets[200:205]}')
        
        
        print(f'Input data looks like: {self.inputs.shape}')
        print(f'Target data looks like: {self.targets.shape}')
        
        
        # split into training and test data    
        # self.training_indices, self.test_indices = NeuralnetTester.get_randomized_training_and_test_indices(len(self.inputs))
        no_test_images = 10
        self.training_indices = [i for i in range(no_test_images, len(self.inputs))]
        self.test_indices = [i for i in range(0, no_test_images)]
        print(f'Splitted into training and test data with: {len(self.training_indices)} training images and {len(self.test_indices)} test images')
        
        input_nodes = self.inputs.shape[1] * self.inputs.shape[2]
        print(f'using {input_nodes} input nodes')
        
        # import weights if exist
        if import_weight_path:
            self.nn = CustomNeuralNetwork.import_neural_net(import_weight_path)
        else:      
            self.nn = CustomNeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
            
    @classmethod
    def get_randomized_training_and_test_indices(cls, number_of_records, number_of_types=3, number_of_test_records_per_type=5) -> Tuple[List[int], List[int]]:
        '''Splits the training data into training and test data. Furthermore, it randomizes the order of the training data.
        
        Parameters
        
            number_of_records :int
                The number of elements ("lines") in the whole set of data
            number_of_types :int, optional (default=3)
                The number of different target types (=outcomes)
            number_of_test_records_per_type :int, optional (default=5)
                The number of elements that should be reserved for testing
                
        Returns
        
            (training_indices, test_indices) :Tuple[List[int], List[int]]
                a tuple of two lists containing the indices of the training and test data
        
        '''
        
        
        start = number_of_records // number_of_types
        training_indices = list(range(number_of_records))
        test_indices = [training_indices[start*i:start*i+number_of_test_records_per_type] for i in range(number_of_types)]
        test_indices = set([item for sublist in test_indices for item in sublist])
        training_indices = list(set(training_indices) - test_indices)   
        random.shuffle(training_indices)
        return training_indices, list(test_indices)
    
    def train(self, epochs :int = 1, auto_save_after_training=False, shuffle=False):
        '''Trains the network with the previously seperated training data
        
        Parameters
        
            epochs :int, optional (default=1)
                The number of epochs to train the network with
            auto_save_after_training :bool, optional (default=False)
                Whether to save the network after training or not
            shuffle :bool, optional (default=False)
                Whether to shuffle the training data after each epoch or not
        
        '''
        
        
        # train the network
        print('training started!\n')
        
        for e in range(epochs):
            print(f'\n### Epoch number {e} ###')
            x = 0
            
            if shuffle:
                random.shuffle(self.training_indices)
                
            for i in self.training_indices:
                x += 1
                print(f'--- Epoch {e}, Image number {x} with index {i} ---', end='\r')
                
                values :np.ndarray = (self.inputs[i] / 255.0 * 0.99)
                targets = np.array([self.targets[i]])
                
                self.nn.train(values.flatten(), targets)
                
        
        print('\n\n!!!! training finished !!!')
        
        if auto_save_after_training:
            self.save('network_data/neuralnet.npy')
        
        pass
    
    def save(self, filename):
        '''Saves the network to the given location
        
        Parameters
        
            filename :str
                The path to the file to save the network (=weights) to
        
        '''
        
        print('saving network...')
        self.nn.save(filename)
        print('saved network successfully!')
    
    def test(self) -> float:
        '''Used to test the network with the previously seperated test data
        
        Returns

            success_rate :float
                The success rate of the network [0;1]
        
        '''
        
        
        print("\nlet's test the network..")
    
        print('test_indices: ', self.test_indices)
        
        success = []
        for i in self.test_indices:
            success.append(self.query(i)[1])
        
        success_rate = sum(success) / len(success)
        print('-'*50)
        print(f'success_rate (avg): {sum(success) / len(success)}  (at a total of {len(success)} tries)')
        print('-'*50)
        return success_rate
    
    def test2(self, lower_bound :int = 0, upper_bound :int = 100, show_image :bool = False, output :bool = False):
        '''goes through the set of data and tests the network with each image
        
        Parameters
        
            lower_bound :int, optional (default=0) The index of the first image
            upper_bound :int, optional (default=100) The image of the last image+1
        
        '''
        
        success = []
        delta = []
        for i in range(lower_bound, upper_bound):
            _, d, s = self.query(i, show_image, output)
            success.append(s)
            delta.append(d)
            
        success_rate = sum(success) / len(success)
        avg_delta = sum(delta) / len(delta)
        print('-'*50)
        print(f'success_rate: {success_rate[0]*100} % (at a total of {len(success)} tries)')
        print(f'success_rate (avg delta): {avg_delta}  (at a total of {len(success)} tries)')
        print('-'*50)
    
    def query(self, query_index :int, show_image :bool = False, output :bool = True) -> bool:
        '''used to query the network with a single image
        
        Parameters
        
            query_index :int
                index of the image to query (= the index of the image in the whole set of images)
                
        Returns

            result: bool
                Whether the network predicted correctly or not
        
        '''
        
        t1 = time.time()
        outputs :np.ndarray = self.nn.query(self.inputs[query_index].flatten())
        print(f'time: {time.time() - t1}')
        
        actual_index = outputs.argmax()
        target = self.targets[query_index]
        
        success = ((target > 0.5) == (outputs[actual_index] > 0.5))
        success = success or (abs(target - outputs[actual_index]) < 0.2)
        
        if output:
            print('.'*30)
            print(f'[{query_index}]:')
            print(f'\tactual: {outputs[actual_index]} \t\t -> rather {"right" if outputs[actual_index] > 0.5 else "left"}')
            print(f'\tshould be: {target} \t\t -> rather {"right" if target > 0.5 else "left"}')
            print(f'\tΔ = {abs(target - outputs[actual_index])} \n')
        
        if show_image:
            cv2.imshow('Image', self.inputs[query_index])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        
        return outputs[actual_index], abs(target - outputs[actual_index]), success
     
    @classmethod           
    def manipulate_images(cls, path_in, path_out):
        utils.down_scale(path_in, path_out)  
        
    def load_data(self, path :str, scale :float = 0.4):
        
        # inputs, targets = utils.load_training_data(path)
        self.sets = utils.load_training_data(path, scale)
        
        # self.inputs = self.sets[0][0][0]
        # self.targets = self.sets[0][0][1]
        self.inputs = []
        self.targets = []
        for i in range(len(self.sets)):
            self.inputs += self.sets[i][0][0]
            self.targets += self.sets[i][0][1]
        
        print(f'inputs shape: {np.array(self.inputs).shape}')
        print(f'targets shape: {np.array(self.targets).shape}')
        
        pass
        
        
    
    
def main():
    # input_dirs = ['TestData/left', 'TestData/right', 'TestData/Straight']
    path = 'Data/data_2/'
    
    # use this one for testing only
    nnt = NeuralnetTester(path_to_data=path, hidden_nodes=3000, output_nodes=1, learning_rate=0.001, image_scale=0.4, import_weight_path="network_data/nn-01.npy")
    # # nnt.query(5)
    # # nnt.test()
    # nnt.test2(0,50, show_image=True, output=True)
    
    # use this one for training (and maybe testing)
    ### deprecated ### nnt = NeuralnetTester(paths_to_input_directories=input_dirs, targets=angles.ANGLES, hidden_nodes=3000, output_nodes=3, learning_rate=0.2)
    # nnt = NeuralnetTester(path_to_data=path, hidden_nodes=3000, output_nodes=1, learning_rate=0.001, image_scale=0.4)
    # nnt.train(epochs=1, auto_save_after_training=False, shuffle=True)
    # nnt.save('network_data/nn-01.npy')
    # nnt.test2(0,3, show_image=False, output=False)
    # nnt.test2(0,10, show_image=False, output=False)
    
    np.dot(nnt.nn.w_input_hidden, nnt.inputs[0].flatten())
    nnt.query(0)
    nnt.query(1)
    
    nnt.query(2)
    
    
    
    # TODO: run again, same settings!!!!!!!!!!!!!!!!!!
    
    pass
    
    
if __name__ == '__main__':
    main()