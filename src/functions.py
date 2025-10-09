import numpy as np

def rosenbrock(x: np.ndarray) -> float:
    """
    計算 Rosenbrock 函數，這是一個非凸函數，常用於優化演算法的性能測試。

    數學公式為： f(x) = Σ[100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]

    Args:
        x (np.ndarray): 一維的變數 numpy 陣列。

    Returns:
        float: Rosenbrock 函數的計算結果。
    """
    result = np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
    return result

def rastrigin(x: np.ndarray) -> float:
    """
    計算 Rastrigin 函數，這是一個高度多模態的函數，具有許多局部最小值，常用於測試演算法。

    數學公式為： f(x) = 10*D + Σ[x_i^2 - 10*cos(2*π*x_i)]

    Args:
        x (np.ndarray): 一維的變數 numpy 陣列。

    Returns:
        float: Rastrigin 函數的計算結果。
    """
    D = len(x)
    sum_term = np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    result = 10 * D + sum_term
    return result

def griewank(x: np.ndarray) -> float:
    """
    計算 Griewank 函數，其特點是擁有廣泛分佈的局部最小值。

    數學公式為： f(x) = 1 + (1/4000) * Σ(x_i^2) - Π(cos(x_i / sqrt(i+1)))

    Args:
        x (np.ndarray): 一維的變數 numpy 陣列。

    Returns:
        float: Griewank 函數的計算結果。
    """
    sum_term = np.sum(x**2) / 4000.0
    # 陣列索引從 0 開始，但數學公式的 i 從 1 開始，所以使用 np.arange(1, len(x) + 1)
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    result = 1 + sum_term - prod_term
    return result