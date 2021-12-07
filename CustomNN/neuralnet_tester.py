from typing import *

import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import utils, angles
import random

from custom_neural_network import CustomNeuralNetwork

# random.seed(1)
# np.random.seed(1)

class NeuralnetTester:
    def __init__(self, inputs=None, targets=None, paths_to_input_directories :List[str] = None, path_to_input_file :str = None, path_to_target_file :str = None, import_weight_path :str = None, input_nodes :int = 1200, hidden_nodes :int = 100, output_nodes :int = 3, learning_rate :float = 0.2):
        
        # import weights if exist
        if import_weight_path:
            self.nn = CustomNeuralNetwork.import_neural_net(import_weight_path)
        else:      
            self.nn = CustomNeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
            
        # prepare input data (training + test data)
        self.inputs = inputs
        self.targets = targets
        
        if paths_to_input_directories:
            _inputs = []
            for a_dir in paths_to_input_directories:
                images = utils.get_images(a_dir)
                _inputs.extend(images)
            self.inputs = _inputs
            
        self.inputs = np.array(self.inputs)
        self.targets = np.array(self.targets)
            
        self.training_indices, self.test_indices = NeuralnetTester.get_training_and_test_indices(len(self.inputs))
            
    @classmethod
    def get_training_and_test_indices(cls, number_of_records, number_of_types=3, number_of_test_records_per_type=5):
        start = number_of_records // number_of_types
        training_indices = list(range(number_of_records))
        test_indices = [training_indices[start*i:start*i+number_of_test_records_per_type] for i in range(number_of_types)]
        test_indices = set([item for sublist in test_indices for item in sublist])
        training_indices = list(set(training_indices) - test_indices)   
        random.shuffle(training_indices)
        return training_indices, list(test_indices)
    
    def train(self, epochs :int = 1, auto_save_after_training=False):
        pass
    
    def save(self, filename):
        pass
    
    def test(self):
        pass
    
    def query(self, query):
        pass
                
            
        
        
    
    
def main():
    input_dirs = ['Testdaten/left', 'Testdaten/right', 'Testdaten/Straight']
    
    nnt = NeuralnetTester(paths_to_input_directories=input_dirs, targets=utils.ANGLES, import_weight_path="network_data/neuralnet.npy")
    
    pass
    
    
if __name__ == '__main__':
    main()