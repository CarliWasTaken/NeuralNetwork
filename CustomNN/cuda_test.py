import numpy as np
import cupy as cp  
from time import time

def benchmark_processor(arr, func, argument):  
    start_time = time()    
    func(arr, argument) # your argument will be broadcasted into a matrix automatically
    finish_time = time()
    elapsed_time = finish_time - start_time    
    return elapsed_time

# load a matrix to global memory
array_cpu = np.random.randint(0, 255, size=(21000, 21000))

# load the same matrix to GPU memory
array_gpu = cp.asarray(array_cpu) 

# benchmark matrix addition on CPU by using a NumPy addition function
cpu_time = benchmark_processor(array_cpu, np.add, 999)

# you need to run a pilot iteration on a GPU first to compile and cache the function kernel on a GPU
benchmark_processor(array_gpu, cp.add, 1)

# benchmark matrix addition on GPU by using CuPy addition function
gpu_time = benchmark_processor(array_gpu, cp.add, 999)
print(gpu_time)

# determine how much is GPU faster
faster_processor = '??' if gpu_time == 0.0 else (gpu_time - cpu_time) / gpu_time * 100

print(f"CPU time: {cpu_time} seconds\nGPU time: {gpu_time} seconds.\nGPU was {faster_processor} percent faster")