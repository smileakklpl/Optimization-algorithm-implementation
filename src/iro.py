import numpy as np

"""
Improved Random Optimization(Improved RO) 改良版隨機最佳化演算法

各參數的解釋&資料型態:
objective_func (callable): 要最小化的目標函數。
bounds (tuple): 搜尋空間的邊界 (min_bound, max_bound)。
dim (int): 搜尋空間的維度。
max_iter (int): 總迭代次數。

最終輸出:
tuple: 包含 (best_solution -> 最佳解, best_fitness -> 最佳解的表現, fitness_history -> 最佳解的表現之歷史紀錄)
"""

def improved_random_optimization(objective_func, bounds, dim, max_iter):

    # 步驟 1: 初始化位置 x 和偏置項 b
    min_bound, max_bound = bounds
    current_pos_x = np.random.uniform(min_bound, max_bound, dim)
    b = np.zeros(dim)
    current_fitness = objective_func(current_pos_x)
    
    # 用於記錄整個過程中的全域最佳解
    best_solution = current_pos_x
    best_fitness = current_fitness
    
    # 歷史迭代記錄
    fitness_history = []
    fitness_history.append(best_fitness)

    # 迭代迴圈 (step 2 到 6)
    for i in range(max_iter):
        # 步驟 2: 取得隨機向量 dx 並評估前進方向
        sigma = (max_bound - min_bound) / 6.0 # 設定標準差為範圍的1/6，因+-3σ涵蓋99.7%的數值
        dx = np.random.normal(0, sigma, dim) # 使用numpy的正態分佈產生隨機向量，即步長
        
        x_forward = current_pos_x + b + dx
        # 確保更新後的解在限定範圍內
        x_forward = np.clip(x_forward, min_bound, max_bound)# 限制在邊界內，必須>=min_bound 且<=max_bound
        fitness_forward = objective_func(x_forward)

        # 步驟 3: 檢查判定條件，if符合則goto step6， else 則 goto step4
        if fitness_forward < current_fitness:
            current_pos_x = x_forward
            current_fitness = fitness_forward
            b = 0.2*b + 0.4*dx
        else:
            # 步驟 4: 檢查判定條件，if符合則goto step6， else 則 goto step5
            x_backward = current_pos_x + b - dx
            x_backward = np.clip(x_backward, min_bound, max_bound)# 限制在邊界內，必須>=min_bound 且<=max_bound
            fitness_backward = objective_func(x_backward)
            
            if fitness_backward < current_fitness:
                current_pos_x = x_backward
                current_fitness = fitness_backward
                b = b - 0.4*dx
            else:
                # 步驟 5: 若無改進，則將偏置項b減半
                b = 0.5*b
        
        # 與過往表現相比，若新的解更好，則更新最佳解位置
        if current_fitness < best_fitness:
            best_solution = current_pos_x
            best_fitness = current_fitness
            
        # 步驟 6: 檢查停止條件 (由max_itre控制)，並記錄最佳解的歷史數值
        fitness_history.append(best_fitness)

    return best_solution, best_fitness, fitness_history
