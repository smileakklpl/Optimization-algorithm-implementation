# Optimization-algorithm-implementation
Implement four optimization algorithms

使用四種演算法來尋找三個已知測試函數的最小值，並將結果進行比較與分析

## 簡介
演算法包括：
- 隨機優化 (Random Optimization, RO)
- 改良版隨機優化 (Improved Random Optimization, IRO)
- 粒子群優化 (Particle Swarm Optimization, PSO)
- 鯨魚優化演算法 (Whale Optimization Algorithm, WOA)

測試函數包括：
- Rastrigin 函數
- Rosenbrock 函數
- Sphere 函數

## 檔案介紹
- pyproject: Python 專案設定檔
- README.md: 專案說明文件
- src/ro.py: 隨機優化演算法(RO) 
- src/iro.py: 改良版隨機優化演算法(IRO) 
- src/pso.py: 粒子群優化演算法(PSO) 
- src/woa.py: 鯨魚優化演算法 (WOA)
- src/test_functions.py: 三種測試函數
- src/visualization.py: 測試函數視覺化
- src/plot_results.py: 結果繪圖
- main.py: 主程式，執行演算法並比較結果
- results/data/: 儲存演算法執行結果的資料
- results/plots/: 儲存結果圖表

## 使用說明
1. 安裝 Python 3.9 以上版本。
2. 創建虛擬環境
    \\python -m venv venv
    \\source venv/bin/activate  # Linux 或 macOS
    \\venv\Scripts\activate     # Windows
    \\或是選用conda
    \\conda create -n myenv python=3.9
    \\conda activate myenv
3. 安裝所需套件：
   在檔案目錄下執行:
   \\pip install -e .
4. 執行主程式：
   \\python main.py