import numpy as np

def whale_optimization_algorithm(objective_func, bounds, dim, population_size, max_iter):
    """
    實作鯨魚優化演算法 (Whale Optimization Algorithm, WOA)。
    此實作基於課程簡報 (introducction.pdf) 第 24 頁的偽代碼。

    Args:
        objective_func (callable): 要最小化的目標函數。
        bounds (tuple): 搜尋空間的邊界 (min_bound, max_bound)。
        dim (int): 搜尋空間的維度。
        population_size (int): 鯨魚的數量。
        max_iter (int): 總迭代次數。

    Returns:
        tuple: 一個包含 (最佳解, 最佳適應度, 適應度歷史紀錄) 的元組。
    """
    min_bound, max_bound = bounds

    # 步驟 1: 初始化鯨魚族群位置，並找出最佳鯨魚 (X*)
    positions = np.random.uniform(min_bound, max_bound, (population_size, dim))
    fitness = np.array([objective_func(p) for p in positions])
    
    best_whale_index = np.argmin(fitness)
    best_whale_position = positions[best_whale_index]
    best_whale_fitness = fitness[best_whale_index]
    
    fitness_history = [best_whale_fitness]

    # 步驟 2: 主要優化迴圈
    for t in range(max_iter):
        # 參數 'a' 從 2 線性遞減到 0
        a = 2 - t * (2 / max_iter)

        for i in range(population_size):
            # 更新參數 A, C, p, l
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            
            A = 2 * a * r1 - a
            C = 2 * r2
            
            p = np.random.rand()
            l = np.random.uniform(-1, 1, dim)
            b = 1  # 定義螺旋形狀的常數

            if p < 0.5:
                # 利用階段: 收縮包圍 或 探索階段: 尋找獵物
                if np.linalg.norm(A) < 1:
                    # 收縮包圍機制 (利用階段) (p.18, 公式 1)
                    D = np.abs(C * best_whale_position - positions[i])
                    positions[i] = best_whale_position - A * D
                else:
                    # 尋找獵物 (探索階段) (p.23, 公式 1)
                    random_whale_index = np.random.randint(population_size)
                    random_whale_position = positions[random_whale_index]
                    D = np.abs(C * random_whale_position - positions[i])
                    positions[i] = random_whale_position - A * D
            else:
                # 螺旋更新位置 (利用階段) (p.21, 公式 1)
                D_prime = np.abs(best_whale_position - positions[i])
                positions[i] = D_prime * np.exp(b * l) * np.cos(2 * np.pi * l) + best_whale_position

            # 處理邊界，確保鯨魚位置在搜尋空間內
            positions[i] = np.clip(positions[i], min_bound, max_bound)

        # 在更新完所有鯨魚後，評估新位置的適應度並更新最佳鯨魚
        for i in range(population_size):
            current_fitness = objective_func(positions[i])
            if current_fitness < best_whale_fitness:
                best_whale_fitness = current_fitness
                best_whale_position = positions[i]
        
        fitness_history.append(best_whale_fitness)

    # 步驟 3: 回傳找到的最佳解
    return best_whale_position, best_whale_fitness, fitness_history
