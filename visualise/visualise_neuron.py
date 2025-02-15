import numpy as np
from ai import Neuron
from matplotlib import pyplot as plt

def y_values(x_values: list[int]):
    slope = 0.5
    y_intercept = 0.5
    NEURON = Neuron(slope, y_intercept)
    return np.array([NEURON.y(x) for x in x_values])

def show(x_values, y_values):
    plt.plot(x_values, y_values, "r")
    min_value = min(min(x_values), min(y_values))
    max_value = max(max(x_values), max(y_values))
    plt.xlim(min_value-0.1, max_value+0.1)
    plt.ylim(min_value-0.1, max_value+0.1)
    plt.show()

if __name__ == "__main__":
    X_VALUES = [0, 10]
    Y_VALUES = y_values(X_VALUES)
    show(X_VALUES, Y_VALUES)