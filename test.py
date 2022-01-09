#!/usr/bin/env python3
"""
Test script for Neural Network with Backpropagation.

Author: Matthias Priesters
"""

import numpy as np
import neural_network as nn

dummy_weights = [
    np.array([[0.1, 0.2],[0.3, 0.4]]),
    np.array([0.2, 1, -3]),
    np.array([1, 2])
]
net = nn.NeuralNetwork(
    shape=[1, 2, 1, 1],
    activation_function='tanh',
    weights=dummy_weights,
)
X = np.array([[2]])
Y = np.array([1])

# shape=[3, 4, 5, 3, 1],

#X = np.array(
#    [[0.12454, -0.154,    0.874564],
#     [0.52345,  0.5324,   0.12432],
#     [0.042315, 0.00432,  0.98345],
#     [0.623452, 0.45645, -0.1234]]
#)
#Y = np.array(
#    [1, -1, 1, -1],
#)

result = net.predict(X)
print(result)

#res2 = net.fit(X, Y)
#print(res2)
