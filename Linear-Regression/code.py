import numpy as np

# 1. Synthetic Dataset (X: feature, y: target)
X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([5.1, 8.2, 10.9, 14.3, 17.0])
N = len(X) # Number of data points

# 2. Hyperparameters and Initial Parameters
learning_rate = 0.01
epochs = 50 # Number of passes over the entire dataset
m = 0.0 # Initialize slope (weight)
c = 0.0 # Initialize intercept (bias)

# 3. SGD Training Loop
print(f"Starting SGD Training with LR={learning_rate} and Epochs={epochs}\n")
for epoch in range(epochs):
    # Stochastic Gradient Descent: Iterate over *each* data point
    for i in range(N):
        x_i = X[i]
        y_i = y[i]

        # Step A: Calculate the prediction for the single point (i)
        y_predicted_i = m * x_i + c

        # Step B: Calculate the partial derivatives (gradients) for the single point (i)
        error = y_i - y_predicted_i
        dJ_dm = -error * x_i
        dJ_dc = -error

        # Step C: Update parameters using the SGD rule
        m = m - learning_rate * dJ_dm
        c = c - learning_rate * dJ_dc
    
    # Optional: Print loss and parameters every few epochs
    if (epoch + 1) % 10 == 0:
        # Calculate overall Mean Squared Error (Loss) for the whole dataset
        y_full_predicted = m * X + c
        mse_loss = np.mean((y - y_full_predicted)**2)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {mse_loss:.4f} | m: {m:.4f}, c: {c:.4f}")

# 4. Final Results
print("\n--- Training Complete ---")
print(f"Final Parameters: Slope (m) = {m:.4f}, Intercept (c) = {c:.4f}")

# 5. Testing with an example
test_x = 6.0
test_y_pred = m * test_x + c
print(f"Prediction for X={test_x}: Y_pred = {test_y_pred:.4f}")