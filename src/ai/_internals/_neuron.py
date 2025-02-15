import numpy as np

class Neuron:
    def __init__(self, slope: np.float16, y_intercept: np.float16):
        self.slope = slope
        self.y_intercept = y_intercept

    def y(self, x: np.float16) -> np.float16:
        return self.slope * x + self.y_intercept

    def x(self, y: np.float16) -> np.float16:
        return (y - self.y_intercept) / self.slope

    def adjust(self, slope_adjustment: np.float16, y_intercept_adjustment: np.float16):
        self.slope += slope_adjustment
        self.y_intercept += y_intercept_adjustment
