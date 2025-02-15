import random
import numpy as np
from ai import Neuron, Model
from matplotlib import pyplot as plt


def show(x_values, y_values, initial_y_pred, final_y_pred, initial_mse, final_mse):
    """Visualize the data points and regression lines."""

    plt.scatter(x_values, y_values, color="blue", label="Data points")
    plt.plot(x_values, initial_y_pred, color="red", label="Initial regression line")
    plt.plot(x_values, final_y_pred, color="green", label="Final regression line")

    # Display MSE values
    print(f"Initial MSE: {initial_mse}, Final MSE: {final_mse}")

    # Add labels and legend
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.legend()
    plt.show()


def test_gradient_descent(input_data: np.ndarray[np.float16]):
    NEURON = Neuron()
    MODEL = Model(input_data, NEURON)

    # Extract x and y values from input_data
    x_values = [x for x, _ in input_data]
    y_values = [y for _, y in input_data]

    initial_y_pred = [NEURON.y(x) for x in x_values]
    initial_mse = MODEL.mse()
    MODEL.fit(learning_rate=0.008, iterations=10_000)
    final_y_pred = [NEURON.y(x) for x in x_values]
    final_mse = MODEL.mse()

    # Visualize the data and regression lines
    show(x_values, y_values, initial_y_pred, final_y_pred, initial_mse, final_mse)


if __name__ == "__main__":
    INPUT_DATA = np.array([(i, i * 2) for i in range(10)], dtype=np.float16)
    test_gradient_descent(INPUT_DATA)
