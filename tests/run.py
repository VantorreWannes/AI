import numpy as np

# Sample data (input x, target y)
x = np.array([1, 2, 3, 4])  # Inputs
y = np.array([2, 4, 6, 8])  # Targets (y = 2x + 0)

# Initialize weights and bias
w = 0.5  # Initial weight
b = 0.1  # Initial bias
learning_rate = 0.01
epochs = 1000

# Training loop
for epoch in range(epochs):
    # Forward pass: Compute predictions
    y_pred = w * x + b
    
    # Compute loss (mean squared error)
    loss = np.mean((y_pred - y) ** 2)
    
    # Backward pass: Compute gradients
    dw = np.mean(2 * (y_pred - y) * x)  # Gradient of loss w.r.t. w
    db = np.mean(2 * (y_pred - y))      # Gradient of loss w.r.t. b
    
    # Update weights and bias
    w -= learning_rate * dw
    b -= learning_rate * db
    
    # Print progress
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}, w: {w:.4f}, b: {b:.4f}")

# Test the trained model
test_x = 5
predicted_y = w * test_x + b
print(f"\nPrediction for x={test_x}: {predicted_y:.2f} (Expected: 10)")