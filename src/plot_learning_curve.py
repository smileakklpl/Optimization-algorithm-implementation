import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# 設定要繪製的函式、維度和演算法
function = ["Rosenbrock", "Rastrigin", "Griewank"]
dims = [2, 10, 30]
algorithm = ["RO", "Improved_RO", "PSO", "WOA"]

def plot_learning_curves():

    output_dir = 'results/plots'
    
    for func_name in function:
        for dim in dims:
            plt.figure(figsize=(10, 7))
            
            for algo_name in algorithm:
                lc_filename = f"results/data/lc_{func_name}_{algo_name}_d{dim}.csv"
                # 讀取 CSV檔案，且跳過標頭
                history = np.loadtxt(lc_filename, delimiter=',', skiprows=1)
                plt.plot(history, label=algo_name)

            plt.title(f'Learning Curves for {func_name} (D={dim})')
            plt.xlabel('Iteration')
            plt.ylabel('Best Fitness (log scale)')
            plt.yscale('log')
            plt.legend()
            plt.grid(True, which="both", ls="--")
            
            plot_filename = os.path.join(output_dir, f"learning_curve_{func_name}_d{dim}.png")
            plt.savefig(plot_filename)
            plt.close()
            print(f"已儲存學習曲線圖: {plot_filename}")

if __name__ == "__main__":
    plot_learning_curves()
    print("執行完畢")
