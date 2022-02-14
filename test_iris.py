#!/usr/bin/env python3
"""
Test script for Neural Network with Backpropagation.
Uses data from the Iris dataset for evaluation.
"""

# Author: Matthias Priesters

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
import neural_network as nn

iris = load_iris()
X = iris['data']
y = None
act = 'sigmoid'
eta = None
comp = 0

# Y is a one-hot encoded version of target
tgt = iris['target']
if act == 'tanh':
    eta = 0.0005
    y = np.ones((len(tgt), 3))
    y *= -1
    comp = 0
elif act == 'sigmoid':
    eta = 0.1
    y = np.zeros((len(tgt), 3))
    comp = 0.5
elif act in ['relu', 'lrelu']:
    eta = 0.01
    y = np.zeros((len(tgt), 3))
    comp = 0.5
for t in range(0, len(tgt)):
    y[t][tgt[t]] = 1.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

shape = [
    [len(X_train[0])],
    [10, act],
    [len(y_train[0]), act],
]
net = nn.NeuralNetwork(
    shape=shape,
    verbose=True,
)

net.fit(X_train, y_train, epochs=500, eta=eta)

y_trainpred = net.predict(X_train)
y_pred = net.predict(X_test)

print('---------')
print(f'Error in-sample: {mean_squared_error(y_train, y_trainpred)}')
print(f'Error out-of-sample: {mean_squared_error(y_test, y_pred)}')


def relabel(y_vec):
    """Turn multi-element vectors into labels.
    The index of the largest value corresponds to the class label
    according to iris.target."""
    return np.array([np.argmax(v) for v in y_vec])


y_train_lbl = relabel(y_train)
y_test_lbl = relabel(y_test)
y_trainpred_lbl = relabel(y_trainpred)
y_pred_lbl = relabel(y_pred)

print(f'Accuracy in-sample: {accuracy_score(y_train_lbl, y_trainpred_lbl)}')
print(f'Accuracy out-of-sample: {accuracy_score(y_test_lbl, y_pred_lbl)}')

print(f'F1 in-sample: {f1_score(y_train_lbl, y_trainpred_lbl, average=None)}')
print(f'F1 out-of-sample: {f1_score(y_test_lbl, y_pred_lbl, average=None)}')

print(f'Confusion matrix in-sample:\n{confusion_matrix(y_train_lbl, y_trainpred_lbl)}')
print(f'Confusion matrix out-of-sample:\n{confusion_matrix(y_test_lbl, y_pred_lbl)}')
