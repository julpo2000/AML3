from keras.datasets import mnist
from src import *
import numpy as np

seed = 1
rng = np.random.RandomState(seed)

def use_mnist():
    # maybe use validation set
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    input_size = len(x_train[0]) * len(x_train[0][0])
    output_size = 1
    network = Neat(input_size, output_size)

def sin_wave():
    phase = rng.uniform(low=0, high=2*np.pi)
    ampl = rng.uniform(0.1, 5)
    f_randomsine = lambda x : np.sin(x + phase) * ampl

    n_inputs = 10
    x = np.arange(0, 2*np.pi, 2*np.pi/n_inputs)
    y = f_randomsine(x)

    network = Neat(1, 1)
    network.test_network(x, y, -2)

sin_wave()