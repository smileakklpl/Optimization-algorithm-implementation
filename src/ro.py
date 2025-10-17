import numpy as np

def random_optimization(objective_func, bounds, dim, max_iter):
    """
    各參數的解釋&資料型態:
    objective_func (callable): 要最小化的目標函數。
    bounds (tuple): 搜尋空間的邊界 (min_bound, max_bound)。
    dim (int): 搜尋空間的維度。
    max_iter (int): 總迭代次數。

    最終輸出:
    tuple: 包含 (best_solution -> 最佳解, best_fitness -> 最佳解的表現, fitness_history -> 最佳解的表現之歷史紀錄)
    """
    # 步驟 1: 隨機選擇一個初始位置 current_pos_x
    min_bound, max_bound = bounds
    current_pos_x = np.random.uniform(min_bound, max_bound, dim)
    current_fitness = objective_func(current_pos_x)
    
    # 用於記錄整個過程中的全域最佳解
    best_solution = current_pos_x
    best_fitness = current_fitness
    
    # 歷史迭代記錄
    fitness_history = []
    fitness_history.append(best_fitness)

    # 迭代迴圈 (step 2 到 4)
    for i in range(max_iter):
        # 步驟 2: 取得隨機向量 dx
        # sigma是一個超參數，這裡設定為搜尋範圍的一部分
        sigma = (max_bound - min_bound) / 6.0 # 設定標準差為範圍的1/6，因+-3σ涵蓋99.7%的數值
        dx = np.random.normal(0, sigma, dim) # 使用numpy的正態分佈產生隨機向量，即步長
        
        candidate_pos = current_pos_x + dx
        # 確保更新後的解在邊界內
        candidate_pos = np.clip(candidate_pos, min_bound, max_bound) # 限制在邊界內，必須>=min_boung 且<=max_bound
        candidate_fitness = objective_func(candidate_pos)

        # 步驟 3: 如果新的解更好，則更新當前位置
        if candidate_fitness < current_fitness:
            current_pos_x = candidate_pos
            current_fitness = candidate_fitness
        
        # 與過往表現相比，若新的解更好，則更新最佳解位置
        if current_fitness < best_fitness:
            best_solution = current_pos_x
            best_fitness = current_fitness
            
        # 步驟 4: 檢查停止條件 (由max_itre控制)，並記錄最佳解的歷史數值
        fitness_history.append(best_fitness)

    return best_solution, best_fitness, fitness_history