#!/usr/bin/env python3
"""
Test script for Neural Network with Backpropagation.
Uses data from the Iris dataset for evaluation.

Author: Matthias Priesters
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import neural_network as nn

iris = load_iris()
X = iris['data']
y = None
act = 'sigmoid'
k = 50000
eta = None

# Y is a one-hot encoded version of target
tgt = iris['target']
if act == 'tanh':
    eta = 0.0005
    y = np.ones((len(tgt), 3))
    y *= -1
elif act == 'sigmoid':
    eta = 0.35
    y = np.zeros((len(tgt), 3))
elif act == 'relu':
    eta = 0.01
    y = np.zeros((len(tgt), 3))
for t in range(0, len(tgt)):
    y[t][tgt[t]] = 1.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

shape = (4, 10, 3)
net = nn.NeuralNetwork(
    shape=shape,
    activation_function=act,
    eta=eta,
    max_iter=k,
)

net.fit(X_train, y_train)

y_trainpred = net.predict(X_train)
y_pred = net.predict(X_test)

print(f'Error in-sample: {mean_squared_error(y_train, y_trainpred)}')
print(f'Error out-of-sample: {mean_squared_error(y_test, y_pred)}')
