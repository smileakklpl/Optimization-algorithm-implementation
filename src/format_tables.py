import csv
import os

# 1. 讀取 CSV 檔案中的數據
data_path = os.path.join('results', 'data', 'summary_statistics.csv')
try:
    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
except FileNotFoundError:
    print(f"錯誤: '{data_path}' 找不到。請先執行 main.py.")
    exit()

# 2. 定義我們需要的結構
ALGORITHMS = ["RO", "Improved_RO", "PSO", "WOA"]
FUNCTIONS = ["Rosenbrock", "Rastrigin", "Griewank"]
STATS_MAP = {
    'Best': '最好',
    'Mean': '平均值',
    'Worst': '最差',
    'StdDev': '標準差'
}

# 3. 為每個演算法處理並印出一個表格
for algo_name in ALGORITHMS:
    print(f"### 表格：{algo_name} 演算法結果\n")
    
    # 表頭
    header = f"| {'Function / Stat':<18} | {'D=2':<12} | {'D=10':<12} | {'D=30':<12} |"
    separator = f"|{'-'*20}|{'-'*14}|{'-'*14}|{'-'*14}|"
    print(header)
    print(separator)
    
    # 表格內容
    for func_name in FUNCTIONS:
        print(f"| **{func_name}**{' ':<10}| {' ':<12} | {' ':<12} | {' ':<12} |")
        for stat_key, stat_name_ch in STATS_MAP.items():
            
            # 尋找對應的數據點
            d2_data = next((item for item in data if item['Algorithm'] == algo_name and item['Function'] == func_name and item['Dimension'] == '2'), None)
            d10_data = next((item for item in data if item['Algorithm'] == algo_name and item['Function'] == func_name and item['Dimension'] == '10'), None)
            d30_data = next((item for item in data if item['Algorithm'] == algo_name and item['Function'] == func_name and item['Dimension'] == '30'), None)

            # 取得數值，如果找不到則為 'N/A'
            val_d2 = d2_data[stat_key] if d2_data else 'N/A'
            val_d10 = d10_data[stat_key] if d10_data else 'N/A'
            val_d30 = d30_data[stat_key] if d30_data else 'N/A'
            
            # 印出統計數據行
            print(f"|  - {stat_name_ch:<16} | {val_d2:<12} | {val_d10:<12} | {val_d30:<12} |")
            
    print("\n") # 每個表格後增加間距
