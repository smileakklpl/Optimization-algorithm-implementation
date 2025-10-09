import numpy as np

def random_optimization(objective_func, bounds, dim, max_iter):
    """
    實作基本隨機優化 (Primitive RO) 演算法。
    此實作基於課程簡報 (introducction.pdf) 第 6 頁的偽代碼。

    Args:
        objective_func (callable): 要最小化的目標函數。
        bounds (tuple): 搜尋空間的邊界 (min_bound, max_bound)。
        dim (int): 搜尋空間的維度。
        max_iter (int): 總迭代次數。

    Returns:
        tuple: 一個包含 (最佳解, 最佳適應度, 適應度歷史紀錄) 的元組。
    """
    # 步驟 1: 隨機選擇一個初始位置 x
    min_bound, max_bound = bounds
    current_pos = np.random.uniform(min_bound, max_bound, dim)
    current_fitness = objective_func(current_pos)
    
    # 用於記錄整個過程中的全域最佳解
    best_solution = current_pos
    best_fitness = current_fitness
    
    fitness_history = [best_fitness]

    # 迭代迴圈 (步驟 2 到 4)
    for _ in range(max_iter):
        # 步驟 2: 取得隨機向量 dx
        # sigma (標準差) 是一個超參數，這裡設定為搜尋範圍的一部分
        sigma = (max_bound - min_bound) * 0.1
        dx = np.random.normal(0, sigma, dim)
        
        candidate_pos = current_pos + dx
        # 確保候選解在邊界內
        candidate_pos = np.clip(candidate_pos, min_bound, max_bound)
        candidate_fitness = objective_func(candidate_pos)

        # 步驟 3: 如果新的解更好，則更新當前位置
        if candidate_fitness < current_fitness:
            current_pos = candidate_pos
            current_fitness = candidate_fitness
        
        # 更新全域最佳解
        if current_fitness < best_fitness:
            best_solution = current_pos
            best_fitness = current_fitness
            
        # 步驟 4: 檢查停止條件 (由 for 迴圈處理)，並記錄歷史
        fitness_history.append(best_fitness)

    return best_solution, best_fitness, fitness_history


def improved_random_optimization(objective_func, bounds, dim, max_iter):
    """
    實作改良式隨機優化 (Improved RO) 演算法。
    此實作基於課程簡報 (introducction.pdf) 第 7 頁的偽代碼。

    Args:
        objective_func (callable): 要最小化的目標函數。
        bounds (tuple): 搜尋空間的邊界 (min_bound, max_bound)。
        dim (int): 搜尋空間的維度。
        max_iter (int): 總迭代次數。

    Returns:
        tuple: 一個包含 (最佳解, 最佳適應度, 適應度歷史紀錄) 的元組。
    """
    # 步驟 1: 初始化位置 x 和偏置項 b
    min_bound, max_bound = bounds
    x = np.random.uniform(min_bound, max_bound, dim)
    b = np.zeros(dim)
    current_fitness = objective_func(x)
    
    best_solution = x
    best_fitness = current_fitness
    
    fitness_history = [best_fitness]

    for _ in range(max_iter):
        # 步驟 2: 取得隨機向量 dx 並評估前進方向
        sigma = (max_bound - min_bound) * 0.1
        dx = np.random.normal(0, sigma, dim)
        
        x_forward = x + b + dx
        x_forward = np.clip(x_forward, min_bound, max_bound)
        fitness_forward = objective_func(x_forward)

        # 步驟 3: 檢查前進方向
        if fitness_forward < current_fitness:
            x = x_forward
            current_fitness = fitness_forward
            b = 0.2 * b + 0.4 * dx
        else:
            # 步驟 4: 檢查後退方向
            x_backward = x + b - dx
            x_backward = np.clip(x_backward, min_bound, max_bound)
            fitness_backward = objective_func(x_backward)
            
            if fitness_backward < current_fitness:
                x = x_backward
                current_fitness = fitness_backward
                b = b - 0.4 * dx
            else:
                # 步驟 5: 若無改進，則衰減偏置項
                b = 0.5 * b
        
        # 更新全域最佳解
        if current_fitness < best_fitness:
            best_solution = x
            best_fitness = current_fitness
            
        # 步驟 6: 檢查停止條件 (由 for 迴圈處理)，並記錄歷史
        fitness_history.append(best_fitness)

    return best_solution, best_fitness, fitness_history
