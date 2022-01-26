#!/usr/bin/env python3
"""
Test script for Neural Network with Backpropagation.
Implements Example 7.1 from Abu-Mostafa et al.: Learning from Data
e-Chapter 7, page 15

Author: Matthias Priesters
"""

from activation_function import vectorize
import numpy as np
import neural_network as nn

shape = (1, 2, 1, 1)
example_weights = [
    np.array([[0.1, 0.2],
              [0.3, 0.4]]),
    vectorize(np.array([0.2, 1, -3])),
    vectorize(np.array([1, 2]))
]
net = nn.NeuralNetwork(
    shape=shape,
    activation_function='tanh',
    weights=example_weights,
    eta=1,
    k=1,
)
X = vectorize(2)
Y = vectorize(1)

net.fit(X, Y)

# compare values with textbook example:
for layer in net.layers:
    print('---------')
    print(f'layer {layer.layer_num}')
    print(f'input:\n{layer.input_values}')
    print(f'signal:\n{layer.signal}')
    print(f'output:\n{layer.activation}')
    print(f'delta:\n{layer.delta}')
    print(f'gradient:\n{layer.input_values @ np.transpose(layer.delta)}')
