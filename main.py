import numpy as np
import os
import csv

# 匯入我們實作的函式與演算法
from src.functions import rosenbrock, rastrigin, griewank
from src.ro import random_optimization
from src.iro import improved_random_optimization
from src.pso import particle_swarm_optimization
from src.woa import whale_optimization_algorithm

# --- 1. 實驗設定 ---

# 要測試的目標函式及其邊界
FUNCTIONS = [
    {"name": "Rosenbrock", "func": rosenbrock, "bounds": (-5, 10)},
    {"name": "Rastrigin", "func": rastrigin, "bounds": (-5.12, 5.12)},
    {"name": "Griewank", "func": griewank, "bounds": (-600, 600)},
]

# 測試維度
DIMS = [2, 10, 30]

# 根據維度設定對應的群體大小和迭代次數
PARAMS_MAP = {
    2: {'swarm_size': 10, 'max_iter': 300},
    10: {'swarm_size': 30, 'max_iter': 1500},
    30: {'swarm_size': 70, 'max_iter': 3000}
}

# 要執行的演算法
ALGORITHMS = [
    {"name": "RO", "func": random_optimization},
    {"name": "Improved_RO", "func": improved_random_optimization},
    {"name": "PSO", "func": particle_swarm_optimization},
    {"name": "WOA", "func": whale_optimization_algorithm},
]

# 每次實驗的重複執行次數
NUM_RUNS = 15

def main():
    # 確保結果資料夾存在
    os.makedirs('results/data', exist_ok=True)

    # 用於儲存所有統計結果的列表
    all_stats = []

    # --- 2. 主要實驗迴圈 ---
    for func_info in FUNCTIONS:
        for dim in DIMS:
            params = PARAMS_MAP[dim]
            max_iter = params['max_iter']
            swarm_size = params['swarm_size']
            bounds = func_info['bounds']
            objective_func = func_info['func']

            for algo_info in ALGORITHMS:
                print(f"執行中... 函式: {func_info['name']}, 維度: {dim}, 演算法: {algo_info['name']}")
                
                run_fitness_results = []
                run_histories = []

                for run in range(NUM_RUNS):
                    print(f"  - 第 {run + 1}/{NUM_RUNS} 次執行")
                    
                    # 根據演算法名稱傳入不同參數
                    if algo_info['name'] in ['PSO', 'WOA']:
                        _, best_fit, history = algo_info['func'](objective_func, bounds, dim, swarm_size, max_iter)
                    else: # RO and Improved_RO
                        _, best_fit, history = algo_info['func'](objective_func, bounds, dim, max_iter)
                    
                    run_fitness_results.append(best_fit)
                    run_histories.append(history)
                
                # --- 3. 計算統計數據 ---
                best_run = np.min(run_fitness_results)
                worst_run = np.max(run_fitness_results)
                mean_run = np.mean(run_fitness_results)
                std_run = np.std(run_fitness_results)
                
                stats = {
                    'Function': func_info['name'],
                    'Dimension': dim,
                    'Algorithm': algo_info['name'],
                    'Best': f'{best_run:.4e}',
                    'Mean': f'{mean_run:.4e}',
                    'Worst': f'{worst_run:.4e}',
                    'StdDev': f'{std_run:.4e}'
                }
                all_stats.append(stats)
                
                # --- 4. 處理並儲存學習曲線數據 ---
                avg_history = np.mean(np.array(run_histories), axis=0)
                lc_filename = f"results/data/lc_{func_info['name']}_{algo_info['name']}_d{dim}.csv"
                np.savetxt(lc_filename, avg_history, delimiter=',', header='avg_fitness', comments='')

    # --- 5. 儲存整體統計數據 ---
    stats_filename = 'results/data/summary_statistics.csv'
    print(f"\n所有實驗完成，正在將統計結果寫入 {stats_filename}")
    with open(stats_filename, 'w', newline='') as csvfile:
        if all_stats:
            writer = csv.DictWriter(csvfile, fieldnames=all_stats[0].keys())
            writer.writeheader()
            writer.writerows(all_stats)
    
    print("\n腳本執行完畢。")

if __name__ == "__main__":
    main()
