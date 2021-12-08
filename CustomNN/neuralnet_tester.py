from typing import *

import numpy as np
from numpy.core.defchararray import count
import scipy.special
import matplotlib.pyplot as plt
import utils, angles
import random

from custom_neural_network import CustomNeuralNetwork

# random.seed(1)
# np.random.seed(1)

class NeuralnetTester:
    def __init__(self, inputs=None, targets=None, paths_to_input_directories :List[str] = None, path_to_input_file :str = None, path_to_target_file :str = None, import_weight_path :str = None, input_nodes :int = 1200, hidden_nodes :int = 3000, output_nodes :int = 3, learning_rate :float = 0.2):
        
        '''The wonderful constructor of the NeuralnetTester class
        
        Parameters
        
            inputs :np.ndarray, optional (default=None)
                The input data to the network
                Can only be `None`, if `paths_to_input_directories` is present
            targets :np.ndarray
                The target data to the network
                Should be present
            paths_to_input_directories :List[str], optional (default=None)
                Can be omitted, if `inputs` is present
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
        
            
        # prepare input data (training + test data)
        self.inputs = inputs
        self.targets = np.array(targets)
        
        # load data from directories
        if paths_to_input_directories:
            print('\nloading input data from directories...')
            _inputs = []
            for a_dir in paths_to_input_directories:
                print(f'directory {a_dir}')
                images = utils.get_images(a_dir)
                _inputs.extend(images)
                print(f'found {len(images)} images')
            self.inputs = _inputs
        
        # convert to numpy array    
        self.inputs = np.array(self.inputs)
        self.targets = np.array(self.targets)
        
        print(f'Input data looks like: {self.inputs.shape}')
        print(f'Target data looks like: {self.targets.shape}')
        
        
        # split into training and test data    
        self.training_indices, self.test_indices = NeuralnetTester.get_randomized_training_and_test_indices(len(self.inputs))
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
            for i in self.training_indices:
                x += 1
                print(f'--- Epoch {e}, Image number {x} with index {i} ---', end='\r')
                values :np.ndarray = (self.inputs[i] / 255.0 * 0.99)
                targets = np.zeros(self.nn.o_nodes) + 0.01
                if self.targets[i] < len(targets): # only set target to 0.99 if index is valid
                    targets[int(self.targets[i])] = 0.99
                self.nn.train(values.flatten(), targets)
            if shuffle:
                random.shuffle(self.training_indices)
        
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
            success.append(self.query(i))
        
        success_rate = sum(success) / len(success)
        print('-'*50)
        print(f'success_rate: {success_rate*100} % at a total of {len(success)} tries')
        print('-'*50)
        return success_rate
    
    def query(self, query_index :int) -> bool:
        '''used to query the network with a single image
        
        Parameters
        
            query_index :int
                index of the image to query (= the index of the image in the whole set of images)
                
        Returns

            result: bool
                Whether the network predicted correctly or not
        
        '''
        
        outputs :np.ndarray = self.nn.query(self.inputs[query_index].flatten())
        actual_index = outputs.argmax()
        target_index = self.targets[query_index]
        
        print('.'*30)
        print(f'outputs [{query_index}]: \n', outputs)
        print('\n --> actual_index: ', actual_index)
        print('should be index: ', target_index, '\n')
        
        return actual_index == target_index
                
            
        
        
    
    
def main():
    input_dirs = ['Testdaten/left', 'Testdaten/right', 'Testdaten/Straight']
    
    # use this one for testing only
    # nnt = NeuralnetTester(paths_to_input_directories=input_dirs, targets=angles.ANGLES, import_weight_path="network_data/neuralnet.npy")
    
    # use this one for training (and maybe testing)
    nnt = NeuralnetTester(paths_to_input_directories=input_dirs, targets=angles.ANGLES, hidden_nodes=3000, output_nodes=3, learning_rate=0.2)
    
    nnt.train(epochs=2, auto_save_after_training=True, shuffle=True)
    nnt.test()
    
    pass
    
    
if __name__ == '__main__':
    main()