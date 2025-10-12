# Optimization-algorithm-implementation
Implement four optimization algorithms

使用四種演算法來尋找三個已知測試函數的最小值，並將結果進行比較與分析。


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

## 實作方式
1. 繪製測試函數圖形  
針對三個函數在維度D=2時，繪製出各自的3D立體圖與等高線圖。
2. 實作並執行4種演算法  
尋找三個函數在維度 D=2、10、30 時的最小值。演算法的疊代次數N_t和粒子數量N_swarm需根據維度大小做對應調整，條件如下:
```bash
when D=2,  then N_t=300  and N_swarm=10  
when D=10, then N_t=1500 and N_swarm=30  
when D=30, then N_t=3000 and N_swarm=70
```
3. 呈現結果
- 統計數據：  
  將每項實驗重複 15 次，並計算出最好、平均、最差、標準差四個數值，整理成一個表格 。  
- 學習曲線：  
  繪製一張圖表，呈現演算法在最佳化過程中，其解的品質隨著疊代次數增加而變化的情況 。   
   
## 檔案介紹
- pyproject: Python 專案設定檔
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
   ```bash
    python -m venv venv  
    source .venv/bin/activate  # Linux 或 macOS  
    .venv\Scripts\activate     # Windows
   ```
    或是選用conda  
   ```bash
    conda create -n myenv python=3.9  
    conda activate myenv
   ```
3. 安裝所需套件：
   在檔案目錄下執行:
   ```bash
   pip install -e .
   ```
4. 執行主程式：
   ```bash
   python main.py  
   ```
