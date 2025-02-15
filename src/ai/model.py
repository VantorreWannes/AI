from ai._internals import Neuron
import numpy as np

class Model:
    def __init__(self, input_data: list[tuple[np.float16, np.float16]], neuron: Neuron):
        self.neuron = neuron
        self.x_values = np.array([point[0] for point in input_data])
        self.y_values = np.array([point[1] for point in input_data])

    def mse(self) -> np.float16:
        predicted_y_values = np.array([self.neuron.y(x) for x in self.x_values])
        return np.mean((self.y_values - predicted_y_values)**2)
    
    def fit(self, learning_rate: np.float16, iterations: int):
        """Performs gradient_descent"""
        for _ in range(iterations):
            predicted_y_values = np.array([self.neuron.y(x) for x in self.x_values])
            errors = self.y_values - predicted_y_values

            slope_gradient = -2 * np.mean(errors * self.x_values)
            y_intercept_gradient = -2 * np.mean(errors)

            self.neuron.adjust(
                -learning_rate * slope_gradient, -learning_rate * y_intercept_gradient
            )