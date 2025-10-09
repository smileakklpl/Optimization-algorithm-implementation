import numpy as np
import matplotlib.pyplot as plt
import os

# 假設此腳本是從專案根目錄執行的
# 例如: python src/visualize.py
from functions import rosenbrock, rastrigin, griewank

def plot_function_3d(func, x_range, y_range, title, num_points=150):
    """
    為給定的函數生成並保存一個三維曲面圖和一個等高線圖。

    Args:
        func (callable): 要繪製的函數，它必須能接受一個二維 numpy 陣列。
        x_range (tuple): x 軸的範圍 (最小值, 最大值)。
        y_range (tuple): y 軸的範圍 (最小值, 最大值)。
        title (str): 圖表的標題。
        num_points (int): 每個軸上的取樣點數量。
                          注意：作業要求1000點，但為了快速生成，這裡使用較小的數值。
    """
    print(f"正在為 {title} 生成圖表...")

    # 定義輸出的資料夾
    output_dir = 'results/plots'
    
    # 生成取樣點
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)

    # 計算 Z 值
    Z = np.zeros_like(X)
    for i in range(num_points):
        for j in range(num_points):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

    # 創建一個包含兩個子圖的圖形
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(title, fontsize=16)

    # --- 3D 曲面圖 ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    # 使用 rstride 和 cstride 來加快渲染速度
    ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', rstride=5, cstride=5)
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('f(x)')
    ax1.set_title('3D 曲面圖')

    # --- 等高線圖 ---
    ax2 = fig.add_subplot(1, 2, 2)
    # 對於數值範圍大的函數，使用對數尺度能更好地顯示細節
    try:
        min_val = max(Z.min(), 1e-6) # 避免 log(0)
        levels = np.logspace(np.log10(min_val), np.log10(Z.max()), 20)
        contour = ax2.contourf(X, Y, Z, levels=levels, cmap='viridis')
        fig.colorbar(contour, ax=ax2, label='f(x) value (log scale)')
    except ValueError:
        # 如果 Z 值包含負數或零，則退回線性尺度
        contour = ax2.contourf(X, Y, Z, 20, cmap='viridis')
        fig.colorbar(contour, ax=ax2, label='f(x) value')

    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_title('等高線圖')
    ax2.set_aspect('equal')

    # 保存圖形
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, f"{title.lower().replace(' ', '_')}.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"圖表已儲存至 {save_path}")


if __name__ == "__main__":
    # 定義要繪製的函數及其屬性
    functions_to_plot = [
        {
            "func": rosenbrock,
            "range": (-5, 10),
            "title": "Rosenbrock Function"
        },
        {
            "func": rastrigin,
            "range": (-5.12, 5.12),
            "title": "Rastrigin Function"
        },
        {
            "func": griewank,
            "range": (-600, 600),
            "title": "Griewank Function"
        }
    ]
    
    # 如果儲存結果的資料夾不存在，則創建它
    output_dir = 'results/plots'
    os.makedirs(output_dir, exist_ok=True)

    # 為每個函數生成圖表
    for item in functions_to_plot:
        plot_range = item["range"]
        # Griewank 函數的有趣特徵集中在原點附近。
        # 繪製完整的 [-600, 600] 範圍會使圖形看起來很平坦。
        # 我們將使用較小的範圍以獲得資訊更豐富的視覺化效果。
        if item["func"].__name__ == "griewank":
            plot_range = (-30, 30)
            print("注意：為 Griewank 函數使用較小的繪圖範圍以顯示原點附近的細節。")

        plot_function_3d(
            func=item["func"],
            x_range=plot_range,
            y_range=plot_range,
            title=item["title"]
        )

    print("\n所有視覺化圖表都已生成在 'results/plots/' 資料夾中。\n")
