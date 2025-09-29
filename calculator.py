import math
import numpy as np

def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    if y == 0:
        raise ValueError("Cannot divide by zero")
    return x / y

def power(x, y):
    return x ** y

def sqrt(x):
    if x < 0:
        raise ValueError("Cannot take the square root of a negative number")
    return math.sqrt(x)

def mean_squared_error(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Input lists must have the same length")
    
    n = len(y_true)
    squared_errors = [(y_true[i] - y_pred[i])**2 for i in range(n)]
    mse = sum(squared_errors) / n
    return np.sqrt(mse)