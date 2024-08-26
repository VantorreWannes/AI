from ais._artificial_neuron import Neuron
import numpy as np

class Model:

    def __init__(self, neuron: Neuron, input_data: list[tuple[np.float16, np.float16]]):
        self.neuron = neuron
        self.x_values = np.array([point[0] for point in input_data])
        self.y_values = np.array([point[1] for point in input_data])
    
    def mse(self) -> np.float16:
        predicted_y_values = np.array([self.neuron.y(x) for x in self.x_values])
        return np.mean((self.y_values - predicted_y_values) ** 2)
    
    