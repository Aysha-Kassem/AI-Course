import numpy as np

# 1. Load and Prepare Data from CSV
# The file 'MultipleLR.csv' is assumed to be accessible.
try:
    data = np.genfromtxt('MultipleLR.csv', delimiter=',')
except FileNotFoundError:
    print("Error: 'MultipleLR.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# Separate features (X) and target (y)
# X: all rows, all columns except the last one (Features: X1, X2, X3)
X = data[:, :-1]
# y: all rows, the last column (Target: Y)
y = data[:, -1]

N, num_features = X.shape # N: number of data points, num_features: 3

# 2. Hyperparameters and Initialization
# Lower learning rate is necessary due to the magnitude of the feature values
learning_rate = 0.0001
epochs = 500

# Initialize weights (theta1, theta2, theta3) and bias (theta0) to zero
weights = np.zeros(num_features)
bias = 0.0

print(f"Starting Multiple LR with SGD Training...")
print(f"Features: {num_features}, Learning Rate: {learning_rate}, Epochs: {epochs}\n")

# 3. SGD Training Loop
for epoch in range(epochs):
    # Shuffle indices for true Stochastic Gradient Descent randomness
    indices = np.arange(N)
    np.random.shuffle(indices)

    # Inner Loop: Iterate through each data point
    for i in indices:
        x_i = X[i, :] # Features for a single point
        y_i = y[i]    # Target for a single point

        # Step A: Calculate the prediction (Hypothesis: y_hat = X.W + b)
        y_predicted_i = np.dot(x_i, weights) + bias

        # Step B: Calculate the Error
        error = y_i - y_predicted_i

        # Step C: Calculate Gradients (for a single data point) and Update Parameters
        
        # Gradient for Weights (theta_j): -error * X_i_j
        gradient_weights = -error * x_i
        weights = weights - learning_rate * gradient_weights
        
        # Gradient for Bias (theta0): -error
        gradient_bias = -error
        bias = bias - learning_rate * gradient_bias
    
    # Optional: Log Loss and parameters every 100 epochs
    if (epoch + 1) % 100 == 0:
        # Calculate overall Mean Squared Error (MSE) Loss for the whole dataset for logging
        y_full_predicted = np.dot(X, weights) + bias
        mse_loss = np.mean((y - y_full_predicted)**2)
        
        weights_str = ', '.join([f'{w:.4f}' for w in weights])
        print(f"Epoch {epoch+1}/{epochs} | Loss: {mse_loss:.4f} | Bias(c): {bias:.4f} | Weights(m): [{weights_str}]")

# 4. Final Results
print("\n--- Training Complete ---")
print(f"Final Bias (theta0) = {bias:.4f}")
print(f"Final Weights (theta1, theta2, theta3) = {weights}")

# 5. Model Testing (Prediction Example)
# Example input: [Feature 1, Feature 2, Feature 3]
test_X = np.array([80, 85, 90])
prediction = np.dot(test_X, weights) + bias
print(f"\nPrediction for test input {test_X}: Y_pred = {prediction:.4f}")