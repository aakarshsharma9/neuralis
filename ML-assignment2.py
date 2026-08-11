import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report


def main():
    print("=" * 70)
    print("      TITANIC DATASET PREPROCESSING AND MACHINE LEARNING PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # STEP 1: DATA LOADING & COLUMN SELECTION
    # --------------------------------------------------------------------------
    print("\n[STEP 1] Loading Data and Standardizing Columns...")
    # Load dataset
    df = pd.read_csv("titanic.csv")

    # Convert all column headers to lowercase first as required
    df.columns = df.columns.str.lower()
    print("Updated Columns:", df.columns.tolist())

    # Retain specified columns
    selected_cols = ["survived", "pclass", "sex", "age", "fare", "embarked"]
    df = df[selected_cols].copy()

    # Identify missing values
    print("\nMissing Values Count per Feature:")
    print(df.isnull().sum())

    # --------------------------------------------------------------------------
    # STEP 2: HANDLING MISSING VALUES
    # --------------------------------------------------------------------------
    print("\n[STEP 2] Imputing Missing Values...")

    # Impute 'age' with mean
    age_mean = df["age"].mean()
    df["age"] = df["age"].fillna(age_mean)

    # Impute 'embarked' with mode (most frequent value)
    embarked_mode = df["embarked"].mode()[0]
    df["embarked"] = df["embarked"].fillna(embarked_mode)

    print(
        f"  -> 'age' missing values imputed with mean: {age_mean:.2f}"
    )
    print(
        f"  -> 'embarked' missing values imputed with mode: '{embarked_mode}'"
    )
    print("Remaining Missing Values Check:", df.isnull().sum().to_dict())

    # --------------------------------------------------------------------------
    # STEP 3: CATEGORICAL ENCODING
    # --------------------------------------------------------------------------
    print("\n[STEP 3] Encoding Categorical Variables...")

    # Label Encoding for 'sex' (Binary conversion)
    label_enc = LabelEncoder()
    df["sex"] = label_enc.fit_transform(df["sex"])
    print("  -> 'sex' encoded using LabelEncoder (male/female -> 0/1)")

    # One-Hot Encoding for 'embarked'
    df = pd.get_dummies(
        df, columns=["embarked"], prefix="embarked", drop_first=True, dtype=int
    )
    print("  -> 'embarked' encoded using One-Hot Encoding (dummy variables)")

    print("\nPreprocessed Sample DataFrame:")
    print(df.head())

    # --------------------------------------------------------------------------
    # STEP 4: FEATURE & TARGET SEPARATION & FEATURE SCALING
    # --------------------------------------------------------------------------
    print("\n[STEP 4] Feature Scaling...")

    # Separate features (X) and target variable (y)
    X = df.drop(columns=["survived"])
    y = df["survived"]

    # Scale numerical features ('age' and 'fare') using StandardScaler
    scaler = StandardScaler()
    scaled_features = ["age", "fare"]
    X[scaled_features] = scaler.fit_transform(X[scaled_features])

    print("  -> 'age' and 'fare' scaled to Mean=0 and Variance=1 using StandardScaler")

    # --------------------------------------------------------------------------
    # STEP 5: TRAIN-TEST SPLIT
    # --------------------------------------------------------------------------
    print("\n[STEP 5] Splitting Data into 80% Train and 20% Test...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"  -> X_train shape: {X_train.shape}")
    print(f"  -> X_test shape : {X_test.shape}")

    # --------------------------------------------------------------------------
    # STEP 6: MODEL EVALUATION & CROSS-VALIDATION
    # --------------------------------------------------------------------------
    print(
        "\n[STEP 6] Model Training & 5-Fold Cross-Validation Comparison..."
    )

    # Define 3 ML models for comparison
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "Gradient Boosting Classifier": GradientBoostingClassifier(
            random_state=42
        ),
    }

    results = []

    for name, model in models.items():
        # Perform 5-Fold Cross-Validation on training data
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring="accuracy"
        )

        # Train model on full training set and evaluate on test set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        results.append({
            "Model": name,
            "CV Mean Accuracy": cv_scores.mean(),
            "CV Std Dev": cv_scores.std(),
            "Test Set Accuracy": test_acc,
        })

    # Summary Results Table
    results_df = pd.DataFrame(results)
    print("\nModel Comparison Results:")
    print(results_df.to_string(index=False))

    # Detailed report for best performing model
    best_model_name = results_df.sort_values(
        by="Test Set Accuracy", ascending=False
    ).iloc[0]["Model"]
    print(f"\nDetailed Classification Report for Best Model ({best_model_name}):")
    best_model = models[best_model_name]
    print(classification_report(y_test, best_model.predict(X_test)))


if __name__ == "__main__":
    main()