# given_tests.py
# Test functions provided to students for testing their implementations
# Run this file to verify your implementation: python3 given_tests.py

import numpy as np
import tempfile
import os
import polyfit
import regularize_cv

def round_array(arr, decimals=4):
    """Helper function to round array elements"""
    return np.round(arr, decimals)

def run_polyfit_tests():
    print("=== Testing polyfit.py ===\n")

    # Test 1: feature_matrix with simple input
    print("Test 1: feature_matrix(x=[2], d=2)")
    x = [2]
    d = 2
    X = polyfit.feature_matrix(x, d)
    print(f"Your output: {X}")
    print(f"Expected: [[4, 2, 1]]")
    print()

    # Test 2: feature_matrix shape
    print("Test 2: feature_matrix(x=[1, 2, 3], d=1)")
    x = [1, 2, 3]
    d = 1
    X = polyfit.feature_matrix(x, d)
    print(f"Your output: {X}")
    print(f"Expected: [[1, 1], [2, 1], [3, 1]]")
    print()

    # Test 3: least_squares with simple linear case
    print("Test 3: least_squares for y = 2x + 1")
    X = [[1, 1], [2, 1], [3, 1]]
    y = [3, 5, 7]
    B = polyfit.least_squares(X, y)
    print(f"Your output: {round_array(B, 3)}")
    print(f"Expected: [2. 1.] (approximately)")
    print()

    # Test 4: main function with provided test case
    print("Test 4: main function with poly.txt (degrees=[2, 4])")
    if os.path.exists('poly.txt'):
        degrees = [2, 4]
        paramFits = polyfit.main('poly.txt', degrees)
        print("Your output for d=2:")
        print(round_array(paramFits[0], 3))
        print("Expected: [-1.265 27.028 88.441] (approximately)")
        print()
        print("Your output for d=4:")
        print(round_array(paramFits[1], 3))
        print("Expected: [-0.023  1.756 -0.888 -0.652 99.924] (approximately)")
    else:
        print("poly.txt not found - skipping this test")
    print()


def run_regularize_cv_tests():
    print("\n=== Testing regularize_cv.py ===\n")

    # Test 1: normalize_train
    print("Test 1: normalize_train")
    X_train = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    X_norm, mean, std = regularize_cv.normalize_train(X_train)
    print(f"Mean of normalized data: {round_array(np.mean(X_norm, axis=0), 5)}")
    print(f"Expected: [0. 0.]")
    print(f"Std of normalized data: {round_array(np.std(X_norm, axis=0), 5)}")
    print(f"Expected: [1. 1.]")
    print()

    # Test 2: normalize_test
    print("Test 2: normalize_test")
    X_test = np.array([[7, 8]], dtype=float)
    train_mean = np.array([3, 4], dtype=float)
    train_std = np.array([2, 2], dtype=float)
    X_norm = regularize_cv.normalize_test(X_test, train_mean, train_std)
    print(f"Your output: {round_array(X_norm, 3)}")
    print(f"Expected: [[2. 2.]]")
    print()

    # Test 3: get_lambda_range
    print("Test 3: get_lambda_range")
    lmbda = regularize_cv.get_lambda_range()
    print(f"Length of lambda array: {len(lmbda)}")
    print(f"Expected: 51")
    print(f"First lambda: {round(lmbda[0], 5)}")
    print(f"Expected: 0.1")
    print(f"Last lambda: {round(lmbda[-1], 1)}")
    print(f"Expected: 1000.0")
    print()

    # Test 4: train_model
    print("Test 4: train_model")
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    y = np.array([1, 2, 3], dtype=float)
    model = regularize_cv.train_model(X, y, l=1.0)
    print(f"Model type: {type(model).__name__}")
    print(f"Expected: Ridge")
    print(f"Has predict method: {hasattr(model, 'predict')}")
    print(f"Expected: True")
    print()

    # Test 5: error function
    print("Test 5: error function")
    X = np.array([[1], [2], [3]], dtype=float)
    y = np.array([2, 4, 6], dtype=float)
    model = regularize_cv.train_model(X, y, l=0.1)
    mse = regularize_cv.error(X, y, model)
    print(f"MSE: {round(mse, 5)}")
    print(f"Expected: < 0.01 (should be small for well-fitted model)")
    print(f"MSE is non-negative: {mse >= 0}")
    print(f"Expected: True")
    print()


if __name__ == '__main__':
    run_polyfit_tests()
    run_regularize_cv_tests()
    print("\n=== All tests complete ===")
    print("Compare your outputs with the expected values above.")
    print("Small differences in decimal places are acceptable.")
