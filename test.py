#!/usr/bin/env python3
"""
Test script for Neural Network with Backpropagation.

Author: Matthias Priesters (mpriesters@users.noreply.github.com)
"""

import numpy as np
import neural_network as nn

net = nn.NeuralNetwork(
    shape=[3, 4, 5, 3, 1],
    activation_function='tanh',
)
X = np.array([0.12454, 0.154, 0.874564])

result = net.predict(X)

print(result)
