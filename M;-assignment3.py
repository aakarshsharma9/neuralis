import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
    print("=" * 70)
    print("      BREAST CANCER DATASET: LOGISTIC REGRESSION ML WORKFLOW")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # 1. DATA LOADING
    # --------------------------------------------------------------------------
    print("\n[STEP 1] Loading Dataset...")
    df = pd.read_csv("breast-cancer.csv")

    print(f"Dataset Shape: {df.shape}")
    print("Columns present:", df.columns.tolist())

    # Identify target column (assuming last column 'class')
    target_col = "class"
    X = df.drop(columns=[target_col])
    y_raw = df[target_col]

    # Map target variable to binary integers (1 for recurrence/positive class, 0 for no recurrence)
    # Handles potential string variations in target values
    positive_label = [
        val for val in y_raw.unique() if "recurrence-events" in str(val)
    ][0]
    y = (y_raw == positive_label).astype(int)

    print(
        f"Target distribution (1 = '{positive_label}', 0 = Other):\n{y.value_counts()}"
    )

    # --------------------------------------------------------------------------
    # 2. TRAIN-TEST SPLIT (TO PREVENT DATA LEAKAGE)
    # --------------------------------------------------------------------------
    print("\n[STEP 2] Splitting Data into 80% Train and 20% Test Sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape : {X_test.shape}")

    # --------------------------------------------------------------------------
    # 3. FEATURE PREPROCESSING & SCALING
    # --------------------------------------------------------------------------
    print("\n[STEP 3] Setting up Feature Encoding and StandardScaler Pipeline...")

    # Identify numerical and categorical columns
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    print(f"Numerical Features  : {num_cols}")
    print(f"Categorical Features: {cat_cols}")

    # ColumnTransformer ensures scaling and one-hot encoding are fit ONLY on train set
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                num_cols,
            ),  # Standardize numerical features
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                cat_cols,
            ),  # OHE categorical features
        ]
    )

    # Fit preprocessor on training data and transform train & test
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    # --------------------------------------------------------------------------
    # 4. MODEL TRAINING
    # --------------------------------------------------------------------------
    print("\n[STEP 4] Training Logistic Regression Model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_prep, y_train)
    print("Model training completed successfully.")

    # --------------------------------------------------------------------------
    # 5. MODEL EVALUATION & DELIVERABLES
    # --------------------------------------------------------------------------
    print("\n[STEP 5] Generating Deliverables...")

    # Predictions
    y_pred = model.predict(X_test_prep)
    y_prob = model.predict_proba(X_test_prep)[:, 1]  # Probabilities for positive class

    # --- Deliverable 1: Classification Report ---
    print("\n" + "=" * 50)
    print("1. CLASSIFICATION REPORT")
    print("=" * 50)
    target_names = ["No Recurrence (0)", f"Recurrence/Malignant (1)"]
    print(
        classification_report(
            y_test, y_pred, target_names=target_names, digits=4
        )
    )

    # --- Deliverable 2: Confusion Matrix & ROC/AUC Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    disp_cm = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=target_names
    )
    disp_cm.plot(ax=axes[0], cmap="Blues", values_format="d")
    axes[0].set_title("Confusion Matrix")

    # ROC & AUC Score Visualization
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    axes[1].plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"ROC Curve (AUC = {roc_auc:.4f})",
    )
    axes[1].plot(
        [0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Chance"
    )
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("Receiver Operating Characteristic (ROC) Curve")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    print("2. Displaying Confusion Matrix and ROC Curve Plots...")
    plt.show()

    print(
        f"\nFinal ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}"
    )


if __name__ == "__main__":
    main()