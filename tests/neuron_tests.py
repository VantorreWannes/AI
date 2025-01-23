import numpy as np
from _internals._neuron import Neuron
from matplotlib import pyplot as plt
import random

def test_simple():
    slope = random.random()
    y_intercept = random.random()
    NEURON = Neuron(slope, y_intercept)
    x_values = [0, 1]
    y_values = np.array([NEURON.y(x) for x in x_values])
    show(x_values, y_values)

def show(x_values, y_values):
    plt.plot(x_values, y_values, "r")
    min_value = min(min(x_values), min(y_values))
    max_value = max(max(x_values), max(y_values))
    plt.xlim(min_value-0.1, max_value+0.1)
    plt.ylim(min_value-0.1, max_value+0.1)
    plt.show()

if __name__ == "__main__":
    test_simple()