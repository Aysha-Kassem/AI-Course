# Machine Learning Optimizers and Stochastic Gradient Descent (SGD) Implementation

This repository addresses two core machine learning requirements: listing common optimizers and implementing a Linear Regression model from scratch using Stochastic Gradient Descent (SGD).

## 1. Top 10 Optimizers in Machine Learning 🧠

Optimizers are algorithms used to modify the attributes of the neural network, such as weights and learning rate, to reduce the losses.

| Optimizer | Explanation |
| :--- | :--- |
| **Stochastic Gradient Descent (SGD)** | Updates the parameters using the gradient of the loss on a single, randomly chosen training example. |
| **Mini-Batch Gradient Descent** | Updates the parameters using the gradient of the loss calculated over a small subset (batch) of the training data. |
| **Batch Gradient Descent** | Updates the parameters using the gradient of the loss calculated over the **entire** training dataset. |
| **Momentum** | Accelerates convergence by adding a fraction of the previous update vector to the current update. |
| **Nesterov Accelerated Gradient (NAG)** | A lookahead version of Momentum that computes the gradient with respect to the approximate future position of the parameters. |
| **AdaGrad (Adaptive Gradient)** | Adapts the learning rate individually for each parameter, scaling it inversely proportional to the square root of the historical sum of squared gradients. |
| **RMSprop (Root Mean Square Propagation)** | Uses a moving average of the squared gradients to normalize the learning rate, helping to avoid aggressive rate decay seen in AdaGrad. |
| **Adam (Adaptive Moment Estimation)** | Combines the ideas of Momentum and RMSprop, utilizing moving averages of both the first (mean) and second (uncentered variance) moments of the gradients. |
| **AdaDelta** | An extension of AdaGrad that seeks to reduce its aggressive, monotonically decreasing learning rate by restricting the window of accumulated past squared gradients. |
| **Adamax** | A variant of Adam that uses the $\ell_{\infty}$ norm (maximum) in the update rule for the second moment, often providing more stable updates. |

***

## 2. Linear Regression with SGD from Scratch 🛠️

This section implements a **Linear Regression** model using **Stochastic Gradient Descent (SGD)** purely with `numpy`, without relying on higher-level machine learning libraries like scikit-learn or TensorFlow.

### Theoretical Foundation

The model aims to fit a line $\hat{y} = mX + c$ to the data by minimizing the **Mean Squared Error (MSE)** loss function.

* **Hypothesis:** $\hat{y}_i = m \cdot x_i + c$
* **Update Rule (SGD):** Parameters are updated after processing **each individual data point** ($x_i, y_i$).
    $$m_{new} = m_{old} - \text{Learning\_Rate} \cdot \frac{\partial J}{\partial m}$$
    $$c_{new} = c_{old} - \text{Learning\_Rate} \cdot \frac{\partial J}{\partial c}$$

### Code Implementation (`sgd_linear_regression.py`)

Save the following code into a file named `sgd_linear_regression.py` in your VS Code workspace.

```python
import numpy as np

# 1. Synthetic Dataset Generation
# Model: y = 3x + 2 + noise
X = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) # Feature
y = np.array([5.1, 8.2, 10.9, 14.3, 17.0]) # Target
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
        # Gradient of MSE with respect to m and c for a single data point
        error = y_i - y_predicted_i
        dJ_dm = -error * x_i
        dJ_dc = -error

        # Step C: Update parameters using the SGD rule
        m = m - learning_rate * dJ_dm
        c = c - learning_rate * dJ_dc
    
    # Optional: Print loss and parameters every few epochs to observe convergence
    if (epoch + 1) % 10 == 0:
        # Calculate overall Mean Squared Error (Loss) for the whole dataset for logging
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