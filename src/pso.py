import numpy as np

"""
Particle Swarm Optimization(PSO) 粒子群最佳化演算法

各參數的解釋&資料型態:
objective_func (callable): 要最小化的目標函數。
bounds (tuple): 搜尋空間的邊界 (min_bound, max_bound)。
dim (int): 搜尋空間的維度。
swarm_size (int): 粒子群中的粒子數量。
max_iter (int): 總迭代次數。
w (float): 慣性權重 (Inertia weight)。
c_p (float): 自我認知係數 (Cognitive coefficient)。
c_g (float): 群體認知係數 (Social coefficient)。

最終輸出:
tuple: 包含 (best_solution -> 最佳解, best_fitness -> 最佳解的表現, fitness_history -> 最佳解的表現之歷史紀錄)
"""

def particle_swarm_optimization(objective_func, bounds, dim, swarm_size, max_iter, w=0.8, c_p=2.0, c_g=2.0):

    min_bound, max_bound = bounds
    
    # 步驟 1: 初始化 gbest 和粒子群
    # 初始化粒子位置和速度
    positions = np.random.uniform(min_bound, max_bound, (swarm_size, dim))
    # 設定初始速度為0
    velocities = np.zeros((swarm_size, dim))

    # 為每個粒子初始化 pbest
    pbest_positions = np.copy(positions)
    pbest_fitness = np.array([objective_func(p) for p in pbest_positions]) #執行初始化之粒子迴圈

    # 初始化 gbest 要注意需包含整個粒子群的最佳解
    gbest_index = np.argmin(pbest_fitness)
    gbest_position = pbest_positions[gbest_index]
    gbest_fitness = pbest_fitness[gbest_index]
    
    # 歷史迭代記錄
    fitness_history = []
    fitness_history.append(gbest_fitness)

    # 步驟 2: 主要優化迴圈
    for i in range(max_iter):
        for j in range(swarm_size):

            # 設定隨機係數 r_p 和 r_g (根據維度dim產生介於0到1之間的隨機數)
            r_p = np.random.rand(dim)
            r_g = np.random.rand(dim)
            
            cognitive_component = c_p * r_p * (pbest_positions[j] - positions[j])
            social_component = c_g * r_g * (gbest_position - positions[j])
            
            # 更新速度 (p.11, 根據老師上課提供之簡報的p11 公式1)
            velocities[j] = w * velocities[j] + cognitive_component + social_component
            
            # 更新位置 (p.11, 根據老師上課提供之簡報的p11 公式2)
            positions[j] = positions[j] + velocities[j]
            
            # 限制在範圍內，必須>=min_boung 且<=max_bound
            positions[j] = np.clip(positions[j], min_bound, max_bound)

            # 評估模型表現並更新best 和 gbest ---
            current_fitness = objective_func(positions[j])
            
            # 更新 pbest (這裡gbest的if比pbest的if要內縮，以確保若pbest有更新才會檢查gbest，增加效率)
            if current_fitness < pbest_fitness[j]:
                pbest_fitness[j] = current_fitness
                pbest_positions[j] = positions[j]
                
                # 更新 gbest
                if current_fitness < gbest_fitness:
                    gbest_fitness = current_fitness
                    gbest_position = positions[j]
        
        fitness_history.append(gbest_fitness)

    # 步驟 3: 回傳找到的最佳解
    return gbest_position, gbest_fitness, fitness_history
