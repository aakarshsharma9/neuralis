import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set seed for reproducibility
np.random.seed(42)


# --- Assignment 1: DataFrame Creation and Indexing ---
def assignment_1():
    print("=== Assignment 1 ===")
    # Task 1: 4 cols, 6 rows random ints. Set index to first column
    df1 = pd.DataFrame(
        np.random.randint(1, 100, size=(6, 4)),
        columns=["Col1", "Col2", "Col3", "Col4"],
    )
    print("Task 1 - Original DataFrame:\n", df1)
    df1_indexed = df1.set_index("Col1")
    print("Task 1 - First Column Set as Index:\n", df1_indexed)

    # Task 2: Cols A, B, C and Index X, Y, Z. Access row Y, col B
    df2 = pd.DataFrame(
        np.random.randint(1, 50, size=(3, 3)),
        columns=["A", "B", "C"],
        index=["X", "Y", "Z"],
    )
    elem = df2.loc["Y", "B"]
    print("Task 2 - DataFrame:\n", df2)
    print(f"Task 2 - Element at Row 'Y' and Col 'B': {elem}")


# --- Assignment 2: DataFrame Operations ---
def assignment_2():
    print("\n=== Assignment 2 ===")
    # Task 1: 3 cols, 5 rows. New column = product of first two columns
    df1 = pd.DataFrame(
        np.random.randint(1, 20, size=(5, 3)), columns=["A", "B", "C"]
    )
    df1["Product_AB"] = df1["A"] * df1["B"]
    print("Task 1 - DataFrame with Product Column:\n", df1)

    # Task 2: 3 cols, 4 rows. Row-wise and column-wise sum
    df2 = pd.DataFrame(
        np.random.randint(1, 20, size=(4, 3)), columns=["X", "Y", "Z"]
    )
    print("Task 2 - DataFrame:\n", df2)
    print("Task 2 - Row-wise Sum:\n", df2.sum(axis=1))
    print("Task 2 - Column-wise Sum:\n", df2.sum(axis=0))


# --- Assignment 3: Data Cleaning ---
def assignment_3():
    print("\n=== Assignment 3 ===")
    # Task 1: Introduce NaNs, fill with column mean
    df1 = pd.DataFrame(
        np.random.randint(1, 50, size=(5, 3)),
        columns=["A", "B", "C"],
        dtype=float,
    )
    df1.iloc[1, 0] = np.nan
    df1.iloc[3, 2] = np.nan
    print("Task 1 - DataFrame with NaNs:\n", df1)
    df1_filled = df1.fillna(df1.mean())
    print("Task 1 - Filled NaNs with Column Mean:\n", df1_filled)

    # Task 2: Introduce NaNs, drop rows with any NaNs
    df2 = pd.DataFrame(
        np.random.randint(1, 50, size=(6, 4)),
        columns=["W", "X", "Y", "Z"],
        dtype=float,
    )
    df2.iloc[0, 1] = np.nan
    df2.iloc[4, 3] = np.nan
    print("Task 2 - Original DataFrame with NaNs:\n", df2)
    df2_dropped = df2.dropna()
    print("Task 2 - DataFrame after dropna():\n", df2_dropped)


# --- Assignment 4: Data Aggregation ---
def assignment_4():
    print("\n=== Assignment 4 ===")
    # Task 1: Group by Category, sum and mean of Value
    df1 = pd.DataFrame({
        "Category": np.random.choice(["A", "B", "C"], size=10),
        "Value": np.random.randint(10, 100, size=10),
    })
    grouped1 = df1.groupby("Category")["Value"].agg(["sum", "mean"])
    print("Task 1 - Category Sum and Mean:\n", grouped1)

    # Task 2: Group by Category, total Sales
    df2 = pd.DataFrame({
        "Product": [f"P{i}" for i in range(1, 9)],
        "Category": np.random.choice(["Electronics", "Clothing"], size=8),
        "Sales": np.random.randint(100, 1000, size=8),
    })
    grouped2 = df2.groupby("Category")["Sales"].sum()
    print("Task 2 - Total Sales by Category:\n", grouped2)


# --- Assignment 5: Merging DataFrames ---
def assignment_5():
    print("\n=== Assignment 5 ===")
    # Task 1: Merge two DFs on common column
    df_left = pd.DataFrame(
        {"id": [1, 2, 3, 4], "Name": ["Alice", "Bob", "Charlie", "David"]}
    )
    df_right = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "Score": [85, 92, 78, 90],
    })
    merged = pd.merge(df_left, df_right, on="id")
    print("Task 1 - Merged DataFrame:\n", merged)

    # Task 2: Concatenate along rows and columns
    df_a = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df_b = pd.DataFrame({"C": [5, 6], "D": [7, 8]})
    concat_rows = pd.concat([df_a, df_b], axis=0, ignore_index=True)
    concat_cols = pd.concat([df_a, df_b], axis=1)
    print("Task 2 - Concatenated Row-wise:\n", concat_rows)
    print("Task 2 - Concatenated Column-wise:\n", concat_cols)


# --- Assignment 6: Time Series Analysis ---
def assignment_6():
    print("\n=== Assignment 6 ===")
    # Task 1: Monthly resampling
    dates1 = pd.date_range("2021-01-01", periods=120, freq="D")
    df1 = pd.DataFrame({"Value": np.random.randint(10, 100, size=120)}, index=dates1)
    monthly_mean = df1.resample("ME").mean()
    print("Task 1 - Monthly Resampled Means (First 5 months):\n", monthly_mean.head())

    # Task 2: 7-day rolling mean
    dates2 = pd.date_range("2021-01-01", "2021-12-31", freq="D")
    df2 = pd.DataFrame(
        {"Value": np.random.randint(10, 100, size=len(dates2))}, index=dates2
    )
    df2["7Day_Rolling_Mean"] = df2["Value"].rolling(window=7).mean()
    print("Task 2 - Rolling Mean (Head):\n", df2.head(10))


# --- Assignment 7: MultiIndex DataFrame ---
def assignment_7():
    print("\n=== Assignment 7 ===")
    # Task 1: Basic MultiIndex indexing and slicing
    arrays = [
        ["Store1", "Store1", "Store2", "Store2"],
        ["Q1", "Q2", "Q1", "Q2"],
    ]
    index = pd.MultiIndex.from_arrays(arrays, names=["Store", "Quarter"])
    df1 = pd.DataFrame({"Sales": [100, 120, 90, 110]}, index=index)
    print("Task 1 - MultiIndex DF:\n", df1)
    print("Task 1 - Slice Store1:\n", df1.loc["Store1"])

    # Task 2: MultiIndex Category & SubCategory Sum
    arrays_cat = [
        ["Fruit", "Fruit", "Veg", "Veg"],
        ["Apple", "Banana", "Carrot", "Celery"],
    ]
    index_cat = pd.MultiIndex.from_arrays(
        arrays_cat, names=["Category", "SubCategory"]
    )
    df2 = pd.DataFrame({"Stock": [50, 80, 40, 30]}, index=index_cat)
    print("Task 2 - Sum by Category:\n", df2.groupby(level="Category").sum())


# --- Assignment 8: Pivot Tables ---
def assignment_8():
    print("\n=== Assignment 8 ===")
    # Task 1: Pivot table Date vs Category for Sum of Value
    df1 = pd.DataFrame({
        "Date": ["2021-01-01", "2021-01-01", "2021-01-02", "2021-01-02"],
        "Category": ["A", "B", "A", "B"],
        "Value": [10, 20, 30, 40],
    })
    pivot1 = df1.pivot_table(
        index="Date", columns="Category", values="Value", aggfunc="sum"
    )
    print("Task 1 - Date/Category Pivot Table:\n", pivot1)

    # Task 2: Pivot table Year vs Quarter for Mean Revenue
    df2 = pd.DataFrame({
        "Year": [2021, 2021, 2022, 2022],
        "Quarter": ["Q1", "Q2", "Q1", "Q2"],
        "Revenue": [1000, 1200, 1100, 1300],
    })
    pivot2 = df2.pivot_table(
        index="Year", columns="Quarter", values="Revenue", aggfunc="mean"
    )
    print("Task 2 - Year/Quarter Pivot Table:\n", pivot2)


# --- Assignment 9: Applying Functions ---
def assignment_9():
    print("\n=== Assignment 9 ===")
    # Task 1: Apply custom function to double values
    df1 = pd.DataFrame(
        np.random.randint(1, 10, size=(5, 3)), columns=["A", "B", "C"]
    )
    print("Task 1 - Original:\n", df1)
    df1_doubled = df1.map(lambda x: x * 2)
    print("Task 1 - Doubled Values:\n", df1_doubled)

    # Task 2: Lambda function for row sum
    df2 = pd.DataFrame(
        np.random.randint(1, 10, size=(6, 3)), columns=["X", "Y", "Z"]
    )
    df2["Row_Sum"] = df2.apply(lambda row: row.sum(), axis=1)
    print("Task 2 - DataFrame with Row Sum Column:\n", df2)


# --- Assignment 10: Working with Text Data ---
def assignment_10():
    print("\n=== Assignment 10 ===")
    text_series = pd.Series(["apple", "banana", "cherry", "date", "elderberry"])

    # Task 1: Uppercase
    upper_series = text_series.str.upper()
    print("Task 1 - Uppercase Series:\n", upper_series)

    # Task 2: First 3 characters
    slice_series = text_series.str[:3]
    print("Task 2 - First 3 Characters:\n", slice_series)


# --- Mini-Project: Dataset Analysis (Iris Dataset) ---
def mini_project_iris():
    print("\n=== Mini-Project: Iris Dataset Cleaning, Stats & Visualization ===")

    # Load dataset
    df = sns.load_dataset("iris")

    # 1. Data Cleaning
    print("Checking Missing Values:\n", df.isnull().sum())
    df_clean = df.drop_duplicates()

    # 2. Summary Statistics
    print("\nDataset Summary Statistics:\n", df_clean.describe())

    # 3. Visualization (3-4 Plots)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Pairwise Scatter / Violinplot
    sns.violinplot(
        data=df_clean, x="species", y="sepal_length", ax=axes[0, 0]
    )
    axes[0, 0].set_title("Sepal Length Distribution by Species")

    # Plot 2: Scatter Plot Sepal Width vs Petal Width
    sns.scatterplot(
        data=df_clean,
        x="sepal_width",
        y="petal_width",
        hue="species",
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Sepal Width vs Petal Width")

    # Plot 3: Boxplot of Petal Length
    sns.boxplot(
        data=df_clean, x="species", y="petal_length", ax=axes[1, 0]
    )
    axes[1, 0].set_title("Petal Length by Species")

    # Plot 4: Feature Correlation Heatmap
    sns.heatmap(
        df_clean.drop("species", axis=1).corr(),
        annot=True,
        cmap="coolwarm",
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Feature Correlation Matrix")

    plt.tight_layout()
    plt.show()


# --- Driver Code ---
if __name__ == "__main__":
    assignment_1()
    assignment_2()
    assignment_3()
    assignment_4()
    assignment_5()
    assignment_6()
    assignment_7()
    assignment_8()
    assignment_9()
    assignment_10()
    mini_project_iris()