#!/usr/bin/env python3
"""
Test script for Neural Network with Backpropagation.
Uses data from the Iris dataset for evaluation.

Author: Matthias Priesters
"""

from activation_function import col_vector
from sklearn.datasets import load_iris
import numpy as np
import neural_network as nn

iris = load_iris()
X = iris['data']
Y = None
act = 'sigmoid'
eta = 1
k = 50000

# Y is a one-hot encoded version of target
tgt = iris['target']
if act == 'tanh':
    eta = 0.005
    Y = np.ones((len(tgt), 3))
    Y *= -1
elif act == 'sigmoid':
    eta = 0.75
    Y = np.zeros((len(tgt), 3))
for t in range(0, len(tgt)):
    Y[t][tgt[t]] = 1.0

shape = (4, 10, 3)
net = nn.NeuralNetwork(
    shape=shape,
    activation_function=act,
    eta=eta,
    max_iter=k,
)

net.fit(X, Y)
