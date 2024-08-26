import numpy

class Neuron():

    def __init__(self, slope, y_intercept):
        self.slope = slope
        self.y_intercept = y_intercept

    def y(self, x):
        return self.slope * x + self.y_intercept

    