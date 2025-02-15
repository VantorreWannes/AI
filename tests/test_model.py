from ai import Neuron, Model
import pytest


@pytest.fixture
def neuron():
    slope = 1
    y_intercept = 1
    return Neuron(slope, y_intercept)


@pytest.fixture
def model(neuron: Neuron):
    input_data = [(1, 1), (2, 1), (3, 1)]
    return Model(neuron, input_data)

def test_mse(model: Model):
    assert model.mse().round(1) == 4.7

def test_fit(model: Model):
    model.fit(0.0008, 1_000)
    assert model.mse().round(1) == 0.0
