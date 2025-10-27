# Machine Learning Optimizers and Stochastic Gradient Descent (SGD) Implementation 🚀

This repository contains the solution for two machine learning tasks: listing common optimizers and implementing a Multiple Linear Regression model from scratch using Stochastic Gradient Descent (SGD).

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

## 2. Multiple Linear Regression with SGD from Scratch 🛠️

This section implements a **Multiple Linear Regression** model using **Stochastic Gradient Descent (SGD)** purely with `numpy`, utilizing the data from the `MultipleLR.csv` file..

### Theoretical Foundation

The model aims to fit a hyperplane to the data by minimizing the **Mean Squared Error (MSE)** loss function.

* **Hypothesis (Multiple Features):**  
  \[
  \hat{y}_i = \theta_0 + \theta_1 x_{i1} + \theta_2 x_{i2} + \theta_3 x_{i3}
  \]

  ✅ Vector Form:
  \[
  \hat{y}_i = \mathbf{\theta}^T \mathbf{x}_i
  \]

---

* **Loss Function (MSE):**  
  \[
  J(\theta) = \frac{1}{2N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2
  \]

---

* **Update Rule (SGD):**  
  Parameters are updated after processing **each individual data point** \((x_i, y_i)\).

  ✅ Gradients:
  \[
  \frac{\partial J}{\partial \theta_j} = - (y_i - \hat{y}_i) \cdot x_{ij}
  \]
  \[
  \frac{\partial J}{\partial \theta_0} = - (y_i - \hat{y}_i)
  \]

  ✅ Final Update Equations:
  \[
  \theta_j := \theta_j + \alpha (y_i - \hat{y}_i) x_{ij}
  \]
  \[
  \theta_0 := \theta_0 + \alpha (y_i - \hat{y}_i)
  \]

---

### Code Implementation (`sgd_linear_regression.py`)

This code reads your uploaded CSV file `MultipleLR.csv`, assumes the first three columns are features (X) and the last is the target (Y), and trains the model.

```python
import numpy as np

# Set the file name from the uploaded dataset
file_name = 'MultipleLR.csv - MultipleLR.csv (1).csv'

# 1. Load and Prepare Data from CSV
try:
    data = np.genfromtxt(file_name, delimiter=',')
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# X: all rows, all columns except the last one (Features)
X = data[:, :-1]
# y: all rows, the last column (Target)
y = data[:, -1]

N, num_features = X.shape

# 2. Hyperparameters and Initialization
learning_rate = 0.0001 
epochs = 500

weights = np.zeros(num_features) # Initialize weights (theta1, theta2, theta3)
bias = 0.0 # Initialize bias (theta0)

print(f"Starting Multiple LR with SGD Training...")
print(f"Features: {num_features}, Learning Rate: {learning_rate}, Epochs: {epochs}\n")

# 3. SGD Training Loop
for epoch in range(epochs):
    # Shuffle indices for true Stochastic Gradient Descent randomness
    indices = np.arange(N)
    np.random.shuffle(indices)

    # Stochastic Gradient Descent: Iterate over *each* data point
    for i in indices:
        x_i = X[i, :] # Single point features
        y_i = y[i]    # Single point target

        # Step A: Calculate the prediction (y_hat = X.W + b)
        y_predicted_i = np.dot(x_i, weights) + bias

        # Step B: Calculate the Error
        error = y_i - y_predicted_i

        # Step C: Calculate Gradients and Update Parameters
        
        # Gradient for Weights: -error * X_i_j
        gradient_weights = -error * x_i
        weights = weights - learning_rate * gradient_weights
        
        # Gradient for Bias: -error
        gradient_bias = -error
        bias = bias - learning_rate * gradient_bias
    
    # Optional: Print loss and parameters every 100 epochs to observe convergence
    if (epoch + 1) % 100 == 0:
        # Calculate overall Mean Squared Error (Loss) for the whole dataset for logging
        y_full_predicted = np.dot(X, weights) + bias
        mse_loss = np.mean((y - y_full_predicted)**2)
        
        weights_str = ', '.join([f'{w:.4f}' for w in weights])
        print(f"Epoch {epoch+1}/{epochs} | Loss: {mse_loss:.4f} | Bias(c): {bias:.4f} | Weights(m): [{weights_str}]")

# 4. Final Results
print("\n--- Training Complete ---")
print(f"Final Bias (theta0) = {bias:.4f}")
print(f"Final Weights (theta1, theta2, theta3) = {weights}")

# 5. Testing with an example
test_X = np.array([80, 85, 90])
prediction = np.dot(test_X, weights) + bias
print(f"\nPrediction for test input {test_X}: Y_pred = {prediction:.4f}")
```


### Execution Output
Running the code with the provided data and hyperparameters yields the following final results:
```
--- Training Complete ---
Final Bias (theta0) = -0.1569
Final Weights (theta1, theta2, theta3) = [0.11068336 0.47119735 1.22109065]

Prediction for test input [80 85 90]: Y_pred = 158.6477
```

### Output

✅ Trained model parameters (θ values)  
✅ Prediction results  
✅ Error progression per iteration  

---

### Dataset

The implementation expects a `MultipleLR.csv` formatted like:

```
x1, x2, x3, y
...
```

---



​