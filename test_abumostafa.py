#!/usr/bin/env python3
"""
Test script for Neural Network with Backpropagation.
Implements Example 7.1 from Abu-Mostafa et al.: Learning from Data
e-Chapter 7, page 15

Author: Matthias Priesters
"""

from activation_function import col_vector
import numpy as np
import neural_network as nn

shape = [
    [1],
    [2, 'tanh'],
    [1, 'tanh'],
    [1, 'tanh'],
]
example_weights = [
    np.array([[0.1, 0.2],
              [0.3, 0.4]]),
    col_vector(np.array([0.2, 1, -3])),
    col_vector(np.array([1, 2]))
]
net = nn.NeuralNetwork(
    shape=shape,
    weights=example_weights,
    eta=1,
    verbose=False,
)
X = col_vector(2)
Y = col_vector(1)

net.fit(X, Y, epochs=1)

# compare values with textbook example:
for layer in net.layers:
    print('---------')
    print(f'layer {layer.layer_num}')
    print(f'input:\n{layer.input_values}')
    print(f'signal:\n{layer.signal}')
    print(f'output:\n{layer.activation}')
    print(f'delta:\n{layer.delta}')
    print(f'gradient:\n{layer.input_values @ np.transpose(layer.delta)}')
