import numpy as np
from ais import Neuron, Model
from matplotlib import pyplot as plt


def test_mse():
    slope = 1
    y_intercept = 0
    NEURON = Neuron(slope, y_intercept)
    MODEL = Model(NEURON, [(0, 0), (1, 1)])
    assert MODEL.mse() == 0


if __name__ == "__main__":
    test_mse()