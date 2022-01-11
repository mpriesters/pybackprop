#!/usr/bin/env python3
"""
Test script for Neural Network with Backpropagation.
Implements the example from Abu-Mostafa et al.: Learning from Data
e-Chapter 7, page 15

Author: Matthias Priesters
"""

from activation_function import vectorize
import numpy as np
import neural_network as nn

shape = (1, 2, 1, 1)
dummy_weights = [
    np.array([[0.1, 0.2],
              [0.3, 0.4]]),
    vectorize(np.array([0.2, 1, -3])),
    vectorize(np.array([1, 2]))
]
net = nn.NeuralNetwork(
    shape=shape,
    activation_function='tanh',
    weights=dummy_weights,
    eta=0.1,
    k=10,
)
X = vectorize(2)
Y = vectorize(1)

net.fit(X, Y)
