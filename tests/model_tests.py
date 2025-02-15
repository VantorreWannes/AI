from ai import Model, Neuron
from matplotlib import pyplot as plt
import random

def plot_data_and_regression(x_values, y_values, initial_y_pred, final_y_pred, initial_mse, final_mse):
    """Visualize the data points and regression lines."""
    plt.scatter(x_values, y_values, color='blue', label='Data points')
    plt.plot(x_values, initial_y_pred, color='red', label='Initial regression line')
    plt.plot(x_values, final_y_pred, color='green', label='Final regression line')
    
    # Display MSE values
    print(f"Initial MSE: {initial_mse}, Final MSE: {final_mse}")
    
    # Add labels and legend
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.legend()
    plt.show()

def test_mse():
    slope = 1
    y_intercept = 0
    NEURON = Neuron(slope, y_intercept)
    MODEL = Model(NEURON, [(0, 0), (1, 1)])
    assert MODEL.mse() == 0

def test_gradient_descent():
    slope = random.random()
    y_intercept = random.random()
    NEURON = Neuron(slope, y_intercept)
    input_data = [(i, i*2) for i in range(10)]
    MODEL = Model(NEURON, input_data)

    # Extract x and y values from input_data
    x_values = [x for x, y in input_data]
    y_values = [y for x, y in input_data]

    initial_y_pred = [NEURON.y(x) for x in x_values]
    initial_mse = MODEL.mse()
    MODEL.fit(learning_rate=0.002, iterations=1500)
    final_y_pred = [NEURON.y(x) for x in x_values]
    final_mse = MODEL.mse()

    # Visualize the data and regression lines
    plot_data_and_regression(x_values, y_values, initial_y_pred, final_y_pred, initial_mse, final_mse)

if __name__ == "__main__":
    test_mse()
    test_gradient_descent()