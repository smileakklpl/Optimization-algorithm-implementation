import numpy as np

"""
Whale Optimization Algorithm(WOA) 鯨群最佳化演算法

各參數的解釋&資料型態:
objective_func (callable): 要最小化的目標函數。
bounds (tuple): 搜尋空間的邊界 (min_bound, max_bound)。
dim (int): 搜尋空間的維度。
population_size (int): 鯨魚的數量。
max_iter (int): 總迭代次數。

最終輸出:
tuple: 包含 (best_solution -> 最佳解, best_fitness -> 最佳解的表現, fitness_history -> 最佳解的表現之歷史紀錄)
"""

def whale_optimization_algorithm(objective_func, bounds, dim, population_size, max_iter):

    min_bound, max_bound = bounds

    # 步驟 1: 初始化鯨魚族群位置，並找出最佳鯨魚 (X*)
    positions = np.random.uniform(min_bound, max_bound, (population_size, dim))
    fitness = np.array([objective_func(p) for p in positions]) # 執行初始化之鯨魚粒子迴圈
    
    best_whale_index = np.argmin(fitness)
    best_whale_position = positions[best_whale_index]
    best_whale_fitness = fitness[best_whale_index]
    
    # 歷史迭代記錄
    fitness_history = []
    fitness_history.append(best_whale_fitness)

    # 步驟 2: 鯨魚粒子迴圈
    for t in range(max_iter):
        # 根據老師提供之簡報所要求，參數a從2線性遞減到 0
        a = 2 - t*(2/max_iter)

        # 搜索空間維度迴圈
        for i in range(population_size):
            # 設定參數 r1, r2, A, C, l, 和機率 p
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            
            A = 2*a*r1 - a
            C = 2*r2
            
            p = np.random.rand()
            l = np.random.uniform(-1, 1, dim)
            b = 1  # 定義螺旋形狀的常數

            if p < 0.5:
                # 利用階段: 收縮包圍 or 探索階段: 尋找獵物
                if np.linalg.norm(A) < 1:
                    # 收縮包圍機制 (講義的p.18 公式1)
                    D = np.abs(C * best_whale_position - positions[i])
                    positions[i] = best_whale_position - A*D
                else:
                    # 尋找獵物 (講義的p.23 公式1)
                    random_whale_index = np.random.randint(population_size)
                    random_whale_position = positions[random_whale_index]
                    D = np.abs(C*random_whale_position - positions[i])
                    positions[i] = random_whale_position - A*D
            else:
                # 螺旋更新位置 (講義的p.21, 公式1&2)
                D_prime = np.abs(best_whale_position - positions[i])
                positions[i] = D_prime * np.exp(b*l) * np.cos(2*np.pi*l) + best_whale_position

            # 限制在範圍內，必須>=min_boung 且<=max_bound，確保更新後的鯨魚在範圍內
            positions[i] = np.clip(positions[i], min_bound, max_bound)

        # 在更新完所有鯨魚後，評估新位置的表現並更新鯨魚最佳解
        for i in range(population_size):
            current_fitness = objective_func(positions[i])
            if current_fitness < best_whale_fitness:
                best_whale_fitness = current_fitness
                best_whale_position = positions[i]

        # 步驟 3: 檢查停止條件，依舊是由max_iter控制，並記錄最佳解的歷史數值
        fitness_history.append(best_whale_fitness)

    return best_whale_position, best_whale_fitness, fitness_history