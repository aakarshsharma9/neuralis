import numpy as np
import numpy.ma as ma

# Seed for reproducibility
np.random.seed(42)


# --- Assignment 1: Array Creation and Manipulation ---
def assignment_1():
    print("=== Assignment 1 ===")
    # Task 1: 5x5 array [1, 20], replace 3rd column with 1
    arr1 = np.random.randint(1, 21, size=(5, 5))
    print("Task 1 - Original Array:\n", arr1)
    arr1[:, 2] = 1
    print("Task 1 - 3rd Column Replaced with 1:\n", arr1)

    # Task 2: 4x4 array 1 to 16, replace diagonal with 0
    arr2 = np.arange(1, 17).reshape(4, 4)
    np.fill_diagonal(arr2, 0)
    print("Task 2 - Diagonal Replaced with 0:\n", arr2)


# --- Assignment 2: Array Indexing and Slicing ---
def assignment_2():
    print("\n=== Assignment 2 ===")
    # Task 1: 6x6 array (1 to 36), slice 3rd-5th rows and 2nd-4th cols
    arr1 = np.arange(1, 37).reshape(6, 6)
    sub_arr = arr1[2:5, 1:4]
    print("Task 1 - Sub-array (Rows 3-5, Cols 2-4):\n", sub_arr)

    # Task 2: 5x5 random array, extract border elements
    arr2 = np.random.randint(1, 20, size=(5, 5))
    top = arr2[0, :]
    bottom = arr2[-1, :]
    left = arr2[1:-1, 0]
    right = arr2[1:-1, -1]
    border = np.concatenate([top, right, bottom[::-1], left[::-1]])
    print("Task 2 - Full 5x5 Array:\n", arr2)
    print("Task 2 - Extracted Border Elements:\n", border)


# --- Assignment 3: Array Operations ---
def assignment_3():
    print("\n=== Assignment 3 ===")
    # Task 1: Two 3x4 arrays, perform basic operations
    a = np.random.randint(1, 10, size=(3, 4))
    b = np.random.randint(1, 10, size=(3, 4))
    print("Task 1 - Addition:\n", a + b)
    print("Task 1 - Subtraction:\n", a - b)
    print("Task 1 - Multiplication:\n", a * b)
    print("Task 1 - Division:\n", np.round(a / b, 2))

    # Task 2: 4x4 array (1 to 16), row-wise and column-wise sum
    arr2 = np.arange(1, 17).reshape(4, 4)
    print(
        "Task 2 - Row-wise Sum:",
        arr2.sum(axis=1),
        "| Column-wise Sum:",
        arr2.sum(axis=0),
    )


# --- Assignment 4: Statistical Operations ---
def assignment_4():
    print("\n=== Assignment 4 ===")
    # Task 1: 5x5 random array statistical measures
    arr1 = np.random.randint(1, 50, size=(5, 5))
    print(f"Task 1 - Mean: {np.mean(arr1):.2f}, Median: {np.median(arr1):.2f}")
    print(f"Task 1 - Std Dev: {np.std(arr1):.2f}, Variance: {np.var(arr1):.2f}")

    # Task 2: 3x3 array (1 to 9), normalize
    arr2 = np.arange(1, 10, dtype=float).reshape(3, 3)
    normalized = (arr2 - np.mean(arr2)) / np.std(arr2)
    print("Task 2 - Normalized Array (Mean=0, Std=1):\n", np.round(normalized, 2))


# --- Assignment 5: Broadcasting ---
def assignment_5():
    print("\n=== Assignment 5 ===")
    # Task 1: Add 1D array to each row of 2D array
    matrix_3x3 = np.random.randint(1, 10, size=(3, 3))
    row_vec = np.array([10, 20, 30])
    print("Task 1 - Row-wise Broadcast Addition:\n", matrix_3x3 + row_vec)

    # Task 2: Subtract 1D array from each column of 2D array
    matrix_4x4 = np.random.randint(1, 10, size=(4, 4))
    col_vec = np.array([1, 2, 3, 4]).reshape(4, 1)  # Reshape for column broadcast
    print(
        "Task 2 - Column-wise Broadcast Subtraction:\n", matrix_4x4 - col_vec
    )


# --- Assignment 6: Linear Algebra ---
def assignment_6():
    print("\n=== Assignment 6 ===")
    # Task 1: Determinant, Inverse, Eigenvalues
    mat = np.array([[2, 1, 1], [1, 3, 2], [1, 0, 0]])
    det = np.linalg.det(mat)
    inv = np.linalg.inv(mat)
    eigenvals, _ = np.linalg.eig(mat)
    print(
        f"Task 1 - Determinant: {det:.2f}\nInverse:\n{np.round(inv, 2)}\nEigenvalues: {np.round(eigenvals, 2)}"
    )

    # Task 2: Matrix Multiplication (2x3 @ 3x2)
    a = np.random.randint(1, 5, size=(2, 3))
    b = np.random.randint(1, 5, size=(3, 2))
    print("Task 2 - Matrix Multiplication Result (2x2):\n", a @ b)


# --- Assignment 7: Advanced Array Manipulation ---
def assignment_7():
    print("\n=== Assignment 7 ===")
    # Task 1: Reshape 3x3 (1 to 9) to (1, 9) and (9, 1)
    arr1 = np.arange(1, 10).reshape(3, 3)
    print("Task 1 - Reshaped to (1, 9):\n", arr1.reshape(1, 9))
    print("Task 1 - Reshaped to (9, 1):\n", arr1.reshape(9, 1))

    # Task 2: Flatten 5x5 random array and reshape back
    arr2 = np.random.randint(1, 20, size=(5, 5))
    flattened = arr2.flatten()
    reshaped_back = flattened.reshape(5, 5)
    print(
        "Task 2 - Reconstructed Array Matches Original:",
        np.array_equal(arr2, reshaped_back),
    )


# --- Assignment 8: Fancy Indexing and Boolean Indexing ---
def assignment_8():
    print("\n=== Assignment 8 ===")
    # Task 1: Corner elements of 5x5 array
    arr1 = np.random.randint(10, 99, size=(5, 5))
    corners = arr1[[0, 0, -1, -1], [0, -1, 0, -1]]
    print("Task 1 - Full Array:\n", arr1)
    print("Task 1 - Corner Elements (TL, TR, BL, BR):", corners)

    # Task 2: Set elements > 10 to 10
    arr2 = np.random.randint(1, 20, size=(4, 4))
    print("Task 2 - Original Array:\n", arr2)
    arr2[arr2 > 10] = 10
    print("Task 2 - Elements > 10 Capped to 10:\n", arr2)


# --- Assignment 9: Structured Arrays ---
def assignment_9():
    print("\n=== Assignment 9 ===")
    # Task 1: Structured array (name, age, weight) sorted by age
    dtype_person = [("name", "U20"), ("age", "i4"), ("weight", "f4")]
    people = np.array(
        [("Alice", 25, 55.5), ("Bob", 20, 72.0), ("Charlie", 30, 68.3)],
        dtype=dtype_person,
    )
    sorted_people = np.sort(people, order="age")
    print("Task 1 - Sorted by Age:\n", sorted_people)

    # Task 2: Point array Euclidean Distance
    dtype_point = [("x", "i4"), ("y", "i4")]
    points = np.array([(0, 0), (3, 4), (1, 1)], dtype=dtype_point)
    coords = np.column_stack((points["x"], points["y"]))
    # Calculate pairwise Euclidean distances
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    print("Task 2 - Pairwise Euclidean Distances Matrix:\n", np.round(dist_matrix, 2))


# --- Assignment 10: Masked Arrays ---
def assignment_10():
    print("\n=== Assignment 10 ===")
    # Task 1: Mask elements > 10 and compute sum of unmasked
    arr1 = np.random.randint(1, 20, size=(4, 4))
    masked_arr1 = ma.masked_greater(arr1, 10)
    print("Task 1 - Masked Array (> 10):\n", masked_arr1)
    print("Task 1 - Sum of Unmasked Elements:", masked_arr1.sum())

    # Task 2: Mask diagonal elements and replace with mean of unmasked
    arr2 = np.random.randint(1, 20, size=(3, 3))
    diag_mask = np.eye(3, dtype=bool)
    masked_arr2 = ma.array(arr2, mask=diag_mask)
    unmasked_mean = masked_arr2.mean()
    result = masked_arr2.filled(unmasked_mean)
    print("Task 2 - Original Array:\n", arr2)
    print(
        f"Task 2 - Diagonal Replaced with Unmasked Mean ({unmasked_mean:.2f}):\n",
        np.round(result, 2),
    )


# --- Execute All Assignments ---
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