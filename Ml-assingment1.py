from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ==============================================================================
# 1. CUSTOM LINEAR REGRESSION IMPLEMENTATION (NUMPY FROM SCRATCH)
# ==============================================================================


class CustomLinearRegression:
    """Linear Regression implemented from scratch using NumPy and Gradient Descent.

    Mathematical Foundation:
        - Hypothesis / Model Equation:  y_hat = X * w + b
        - Cost Function (MSE):          L = (1/N) * sum((y_hat - y)^2)
        - Gradient wrt Weights (dw):    dw = (2/N) * X^T * (y_hat - y)
        - Gradient wrt Bias (db):       db = (2/N) * sum(y_hat - y)
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 5000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None  # Model weights vector (slope)
        self.bias = None  # Model bias scalar (intercept)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Trains the linear model using Batch Gradient Descent."""
        n_samples, n_features = X.shape

        # Step 1: Initialize weights to zeros and bias to 0.0
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Step 2: Optimization loop (Gradient Descent)
        for _ in range(self.n_iterations):
            # Compute predicted target values (Forward pass: y_hat = Xw + b)
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute partial derivatives of Mean Squared Error with respect to parameters
            dw = (2 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (2 / n_samples) * np.sum(y_predicted - y)

            # Step 3: Update parameters in the direction of negative gradient
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates predictions for input data matrix X."""
        return np.dot(X, self.weights) + self.bias


# ==============================================================================
# 2. PYTORCH LINEAR REGRESSION MODEL
# ==============================================================================


class PyTorchLinearRegression(nn.Module):
    """Linear Regression implemented as a PyTorch neural network module."""

    def __init__(self, in_features: int):
        super(PyTorchLinearRegression, self).__init__()
        # Single linear layer: computes output = input * weight^T + bias
        self.linear = nn.Linear(in_features=in_features, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward computation graph."""
        return self.linear(x)


# ==============================================================================
# 3. HELPER FUNCTION TO TRAIN PYTORCH MODEL
# ==============================================================================


def train_pytorch_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    in_features: int,
    epochs: int = 5000,
    lr: float = 0.01,
):
    """Converts NumPy arrays to PyTorch Tensors and runs the training loop."""

    # Convert NumPy arrays to PyTorch Tensors (float32 required for PyTorch nn modules)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(
        1
    )  # Reshape (N,) to (N, 1)

    # Instantiate model, loss criterion, and SGD optimizer
    model = PyTorchLinearRegression(in_features=in_features)
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # PyTorch Training Loop
    model.train()
    for epoch in range(epochs):
        # 1. Reset gradients from previous step
        optimizer.zero_grad()

        # 2. Forward pass: compute predicted outputs
        predictions = model(X_train_tensor)

        # 3. Compute loss
        loss = criterion(predictions, y_train_tensor)

        # 4. Backward pass: compute gradients via Backpropagation
        loss.backward()

        # 5. Optimizer step: update model parameters
        optimizer.step()

    return model


# ==============================================================================
# MAIN EXECUTION WITH PRACTICAL EXAMPLES
# ==============================================================================

if __name__ == "__main__":
    print(
        "======================================================================"
    )
    print("      COMPARING LINEAR REGRESSION: SCIKIT-LEARN vs CUSTOM vs PYTORCH  ")
    print(
        "======================================================================\n"
    )

    # --------------------------------------------------------------------------
    # EXAMPLE 1: DATA PREPARATION
    # --------------------------------------------------------------------------
    print("[1] Generating Synthetic Dataset...")
    # Generate 1,000 samples with 3 input features and slight gaussian noise
    X, y = make_regression(
        n_samples=1000, n_features=3, noise=15.0, random_state=42
    )

    # Split into 80% training set and 20% testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"    Train shape: {X_train.shape} | Test shape: {X_test.shape}\n")

    # --------------------------------------------------------------------------
    # EXAMPLE 2: MODEL TRAINING
    # --------------------------------------------------------------------------
    print("[2] Training Models...")

    # A. Scikit-Learn (Closed-Form / OLS Analytical Solution)
    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train, y_train)

    # B. Custom Implementation (NumPy Batch Gradient Descent)
    custom_model = CustomLinearRegression(
        learning_rate=0.01, n_iterations=5000
    )
    custom_model.fit(X_train, y_train)

    # C. PyTorch Implementation (Auto-Differentiation + SGD)
    pytorch_model = train_pytorch_model(
        X_train, y_train, in_features=3, epochs=5000, lr=0.01
    )
    print("    All 3 models trained successfully!\n")

    # --------------------------------------------------------------------------
    # EXAMPLE 3: TEST DATA PREDICTIONS & EVALUATION
    # --------------------------------------------------------------------------
    print("[3] Evaluating Predictions on Test Set...")

    # Predict with Scikit-Learn
    sk_preds = sklearn_model.predict(X_test)

    # Predict with Custom NumPy Model
    custom_preds = custom_model.predict(X_test)

    # Predict with PyTorch Model (Evaluation mode without tracking gradients)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_preds = pytorch_model(X_test_tensor).numpy().flatten()

    # Extract PyTorch learned parameters for comparison
    pt_weights = pytorch_model.linear.weight.detach().numpy().flatten()
    pt_bias = pytorch_model.linear.bias.item()

    # --------------------------------------------------------------------------
    # EXAMPLE 4: PARAMETER & PERFORMANCE COMPARISON TABLE
    # --------------------------------------------------------------------------
    print("[4] Model Parameter & Metric Comparison Table:")

    comparison_df = pd.DataFrame({
        "Parameter / Metric": [
            "Weight 1 (w1)",
            "Weight 2 (w2)",
            "Weight 3 (w3)",
            "Intercept (b)",
            "Test MSE (Lower is better)",
            "Test R2 Score (Closer to 1 is better)",
        ],
        "Scikit-Learn (OLS)": [
            sklearn_model.coef_[0],
            sklearn_model.coef_[1],
            sklearn_model.coef_[2],
            sklearn_model.intercept_,
            mean_squared_error(y_test, sk_preds),
            r2_score(y_test, sk_preds),
        ],
        "Custom (NumPy GD)": [
            custom_model.weights[0],
            custom_model.weights[1],
            custom_model.weights[2],
            custom_model.bias,
            mean_squared_error(y_test, custom_preds),
            r2_score(y_test, custom_preds),
        ],
        "PyTorch (SGD)": [
            pt_weights[0],
            pt_weights[1],
            pt_weights[2],
            pt_bias,
            mean_squared_error(y_test, pytorch_preds),
            r2_score(y_test, pytorch_preds),
        ],
    })

    print(comparison_df.to_string(index=False))
    print("\n")

    # --------------------------------------------------------------------------
    # EXAMPLE 5: DEMONSTRATING INDIVIDUAL PREDICTIONS ON NEW UNSEEN SAMPLES
    # --------------------------------------------------------------------------
    print("[5] Example Predictions on 3 Unseen Samples:")
    print("-" * 70)

    # Take the first 3 samples from the test set as our example inputs
    sample_inputs = X_test[:3]
    actual_targets = y_test[:3]

    for i in range(len(sample_inputs)):
        sample = sample_inputs[i]
        actual = actual_targets[i]

        # Model individual predictions
        pred_sk = sklearn_model.predict([sample])[0]
        pred_custom = custom_model.predict(sample)

        sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred_pt = pytorch_model(sample_tensor).item()

        print(f"Sample #{i + 1} Inputs [x1, x2, x3]: {sample.round(3)}")
        print(f"  -> Actual Ground Truth (y) : {actual:.2f}")
        print(f"  -> Scikit-Learn Prediction : {pred_sk:.2f}")
        print(f"  -> Custom NumPy Prediction  : {pred_custom:.2f}")
        print(f"  -> PyTorch SGD Prediction   : {pred_pt:.2f}")
        print("-" * 70)