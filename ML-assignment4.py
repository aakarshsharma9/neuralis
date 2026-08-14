import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ==============================================================================
# 1. DEFINE THE NEURAL NETWORK (INHERITING FROM torch.nn.Module)
# ==============================================================================
class IrisMLP(nn.Module):
    """Feed-forward Neural Network (Multi-Layer Perceptron) for Iris Classification.

    Architecture:
      - Input Layer  : 4 features (sepal_length, sepal_width, petal_length, petal_width)
      - Hidden Layer : 16 neurons with ReLU activation function
      - Output Layer : 3 neurons (one for each class: setosa, versicolor, virginica)
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 16, output_dim: int = 3):
        super(IrisMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Input to Hidden layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Hidden to Output layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Manual implementation of the forward pass."""
        x = F.relu(self.fc1(x))  # Apply Linear transformation + ReLU activation
        x = self.fc2(
            x
        )  # Raw logits output (CrossEntropyLoss automatically applies Softmax)
        return x


# ==============================================================================
# 2. MAIN EXECUTION PIPELINE
# ==============================================================================
def main():
    print("=" * 70)
    print("         PYTORCH FEED-FORWARD NEURAL NETWORK (IRIS CLASSIFICATION)")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # STEP 1: LOAD DATASET & PREPROCESS
    # --------------------------------------------------------------------------
    print("\n[STEP 1] Loading and Preprocessing Data...")
    df = pd.read_csv("iris.csv")

    # Features (first 4 columns) and Target (species column)
    X = df.iloc[:, :-1].values
    y = df["species"].values

    # Encode string target labels (setosa, versicolor, virginica -> 0, 1, 2)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Standardize numerical features for better neural network convergence
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --------------------------------------------------------------------------
    # STEP 2: TRAIN-TEST SPLIT
    # --------------------------------------------------------------------------
    print("[STEP 2] Splitting Dataset into 80% Train and 20% Test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )

    # --------------------------------------------------------------------------
    # STEP 3: CONVERT DATA TO PYTORCH TENSORS
    # --------------------------------------------------------------------------
    print("[STEP 3] Converting Data into PyTorch Tensors...")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Target must be LongTensor for CrossEntropyLoss

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    print(f"  -> X_train tensor shape: {X_train_tensor.shape}")
    print(f"  -> X_test tensor shape : {X_test_tensor.shape}")

    # --------------------------------------------------------------------------
    # STEP 4: MODEL INITIALIZATION, LOSS FUNCTION & OPTIMIZER
    # --------------------------------------------------------------------------
    print("\n[STEP 4] Initializing Model, Criterion, and Optimizer...")
    model = IrisMLP(input_dim=4, hidden_dim=16, output_dim=3)

    # CrossEntropyLoss expects unnormalized raw logits and class indices as targets
    criterion = nn.CrossEntropyLoss()

    # Adam Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("Model Architecture:")
    print(model)

    # --------------------------------------------------------------------------
    # STEP 5: MODEL TRAINING LOOP
    # --------------------------------------------------------------------------
    print("\n[STEP 5] Training Neural Network...")
    epochs = 150

    model.train()
    for epoch in range(1, epochs + 1):
        # 1. Clear previous gradients
        optimizer.zero_grad()

        # 2. Forward pass: compute predictions
        outputs = model(X_train_tensor)

        # 3. Calculate Loss
        loss = criterion(outputs, y_train_tensor)

        # 4. Backward pass: compute gradients
        loss.backward()

        # 5. Update weights
        optimizer.step()

        # Print training progress every 20 epochs
        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{epochs}] ---> Training Loss: {loss.item():.4f}")

    # --------------------------------------------------------------------------
    # STEP 6: EVALUATION ON TEST DATA
    # --------------------------------------------------------------------------
    print("\n[STEP 6] Evaluating Model on Test Data...")
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Disable gradient computation for testing
        test_outputs = model(X_test_tensor)

        # Get class predictions (index of max logit along dimension 1)
        _, predictions = torch.max(test_outputs, dim=1)

        # Calculate Accuracy
        correct_preds = (predictions == y_test_tensor).sum().item()
        total_samples = y_test_tensor.size(0)
        accuracy = (correct_preds / total_samples) * 100.0

    print("-" * 50)
    print(f"  Total Test Samples   : {total_samples}")
    print(f"  Correct Predictions  : {correct_preds}")
    print(f"  Test Accuracy        : {accuracy:.2f}%")
    print("-" * 50)


if __name__ == "__main__":
    main()