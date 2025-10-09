import numpy as np

def particle_swarm_optimization(objective_func, bounds, dim, swarm_size, max_iter, w=0.8, c_p=2.0, c_g=2.0):
    """
    實作粒子群優化 (Particle Swarm Optimization, PSO) 演算法。

    Args:
        objective_func (callable): 要最小化的目標函數。
        bounds (tuple): 搜尋空間的邊界 (min_bound, max_bound)。
        dim (int): 搜尋空間的維度。
        swarm_size (int): 粒子群中的粒子數量。
        max_iter (int): 總迭代次數。
        w (float): 慣性權重 (Inertia weight)。
        c_p (float): 自我認知係數 (Cognitive coefficient)。
        c_g (float): 群體認知係數 (Social coefficient)。

    Returns:
        tuple: 一個包含 (最佳解, 最佳適應度, 適應度歷史紀錄) 的元組。
    """
    min_bound, max_bound = bounds
    
    # 步驟 1: 初始化 gbest 和粒子群
    # 初始化粒子位置和速度
    positions = np.random.uniform(min_bound, max_bound, (swarm_size, dim))
    # 根據簡報，速度可以初始化為 0 或一個較小的隨機值
    velocities = np.zeros((swarm_size, dim))

    # 為每個粒子初始化 pbest
    pbest_positions = np.copy(positions)
    pbest_fitness = np.array([objective_func(p) for p in pbest_positions])

    # 初始化 gbest
    gbest_index = np.argmin(pbest_fitness)
    gbest_position = pbest_positions[gbest_index]
    gbest_fitness = pbest_fitness[gbest_index]
    
    fitness_history = [gbest_fitness]

    # 步驟 2: 主要優化迴圈
    for _ in range(max_iter):
        for i in range(swarm_size):
            # --- 更新速度 (p.11, 公式 1) ---
            r_p = np.random.rand(dim)
            r_g = np.random.rand(dim)
            
            cognitive_component = c_p * r_p * (pbest_positions[i] - positions[i])
            social_component = c_g * r_g * (gbest_position - positions[i])
            
            velocities[i] = w * velocities[i] + cognitive_component + social_component
            
            # --- 更新位置 (p.11, 公式 2) ---
            positions[i] = positions[i] + velocities[i]
            
            # 處理邊界，確保粒子位置在搜尋空間內
            positions[i] = np.clip(positions[i], min_bound, max_bound)

            # --- 評估適應度並更新 pbest 和 gbest ---
            current_fitness = objective_func(positions[i])
            
            # 更新 pbest
            if current_fitness < pbest_fitness[i]:
                pbest_fitness[i] = current_fitness
                pbest_positions[i] = positions[i]
                
                # 更新 gbest
                if current_fitness < gbest_fitness:
                    gbest_fitness = current_fitness
                    gbest_position = positions[i]
        
        fitness_history.append(gbest_fitness)

    # 步驟 3: 回傳找到的最佳解
    return gbest_position, gbest_fitness, fitness_history
