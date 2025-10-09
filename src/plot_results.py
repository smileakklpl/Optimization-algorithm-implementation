import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# --- 設定 (應與 main.py 保持一致) ---
FUNCTIONS = ["Rosenbrock", "Rastrigin", "Griewank"]
DIMS = [2, 10, 30]
ALGORITHMS = ["RO", "Improved_RO", "PSO", "WOA"]

def generate_markdown_table():
    """讀取彙總的 CSV 檔案並印出格式化的 Markdown 表格。"""
    print("--- 實驗結果統計 ---\n")
    
    try:
        with open('results/data/summary_statistics.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
    except FileNotFoundError:
        print("錯誤: 'results/data/summary_statistics.csv' 找不到。請先執行 main.py ويعود.")
        return

    for func_name in FUNCTIONS:
        print(f"### {func_name} Function\n")
        header = "| Algorithm     | Dimension | Best         | Mean         | Worst        | Std Dev      |"
        separator = "|---------------|-----------|--------------|--------------|--------------|--------------|"
        print(header)
        print(separator)
        
        # 按照演算法和維度順序篩選和排序數據
        for algo_name in ALGORITHMS:
            for dim in DIMS:
                row_data = next((item for item in data if item['Function'] == func_name and item['Algorithm'] == algo_name and int(item['Dimension']) == dim), None)
                if row_data:
                    print(f"| {row_data['Algorithm']:<13} | {row_data['Dimension']:<9} | {row_data['Best']:<12} | {row_data['Mean']:<12} | {row_data['Worst']:<12} | {row_data['StdDev']:<12} |")
        print("\n")


def plot_learning_curves():
    """讀取學習曲線數據並生成比較圖。"""
    print("--- 正在生成學習曲線圖 ---\n")
    output_dir = 'results/plots'
    
    for func_name in FUNCTIONS:
        for dim in DIMS:
            plt.figure(figsize=(10, 7))
            
            for algo_name in ALGORITHMS:
                lc_filename = f"results/data/lc_{func_name}_{algo_name}_d{dim}.csv"
                try:
                    # 讀取 CSV，跳過標頭
                    history = np.loadtxt(lc_filename, delimiter=',', skiprows=1)
                    plt.plot(history, label=algo_name)
                except (FileNotFoundError, IOError):
                    print(f"警告: 找不到或無法讀取學習曲線檔案 {lc_filename} ويعود.")
                    continue

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
    generate_markdown_table()
    plot_learning_curves()
    print("\n結果呈現腳本執行完畢")
