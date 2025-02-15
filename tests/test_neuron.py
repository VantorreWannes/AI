from ai import Neuron
import pytest

@pytest.fixture
def neuron():
    slope = 0.5
    y_intercept = 0.5
    return Neuron(slope, y_intercept)

@pytest.mark.parametrize("x_input,expected_y", [(0, 0.5), (1, 1.0), (2, 1.5)])
def test_y(neuron: Neuron, x_input: int, expected_y: float):
    assert neuron.y(x_input) == expected_y

@pytest.mark.parametrize("y_input,expected_x", [(0.5, 0.0), (1.0, 1.0), (1.5, 2.0)])
def test_x(neuron: Neuron, y_input: float, expected_x: float):
    assert neuron.x(y_input) == expected_x