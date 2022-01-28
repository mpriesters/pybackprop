"""
Activation Functions for a Neural Network with Backpropagation.

Author: Matthias Priesters
"""

from abc import ABC, abstractmethod
import numpy as np
import numbers


def col_vector(x):
    """Turn scalars and single-dimension arrays into column vector arrays."""
    if isinstance(x, (np.number, numbers.Complex)):
        return np.array([[x]])
    elif isinstance(x, np.ndarray) and len(x.shape) == 1:
        return x.reshape(x.shape[0], 1)
    return x


class ActivationFunction(ABC):
    """Abstract Activation Function class."""

    @abstractmethod
    def function(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass


class TanHyp(ActivationFunction):
    """Hyperbolic Tangent Activation Function."""

    def function(self, x):
        res = np.tanh(x)
        return col_vector(res)

    def derivative(self, x):
        res = 1 - (self.function(x))**2
        return col_vector(res)


class Sigmoid(ActivationFunction):
    """Logistic Sigmoid Activation Function."""

    def function(self, x):
        res = 1 / (1 + np.exp(-x))
        return col_vector(res)

    def derivative(self, x):
        # source: https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
        res = self.function(x) * (1 - self.function(x))
        return col_vector(res)


class Relu(ActivationFunction):
    """Rectified Linear Unit Activation Function."""

    def function(self, x):
        res = np.maximum(0, x)
        return col_vector(res)

    def derivative(self, x):
        vec_deriv = np.vectorize(lambda s: 0.0 if s < 0.0 else 1.0)
        return col_vector(vec_deriv(x))
