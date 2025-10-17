import numpy as np
import matplotlib.pyplot as plt
import os
from functions import rosenbrock, rastrigin, griewank

def plot_function_3d(func, x_range, y_range, title, num_points, output_dir):
    """
        各參數的解釋&資料型態:
        func (callable): 要繪製的函數，它必須能接受一個二維 numpy 陣列
        x_range (tuple): x 軸的範圍 (最小值, 最大值)
        y_range (tuple): y 軸的範圍 (最小值, 最大值)
        title (str): 圖表的標題
        num_points (int): 每個軸上的取樣點數量
        output_dir (str): 圖表儲存的目錄
    """
    print(f"正在生成 {title} 的圖表...")
    
    #生成x與y軸的取樣點
    x1 = np.linspace(x_range[0], x_range[1], num_points)
    x2 = np.linspace(y_range[0], y_range[1], num_points)
    x1, x2 = np.meshgrid(x1, x2)

    #計算yk的值
    yk = np.zeros_like(x1)
    for i in range(num_points):
        for j in range(num_points):
            yk[i, j] = func(np.array([x1[i, j], x2[i, j]]))

    #使用plt繪製3D立體圖和等高線圖
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(title, fontsize=16)

    #因plt預設字型不支援中文，故設定字型為YaHei以正確顯示中文標籤
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

    #3D立體圖
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(x1, x2, yk, cmap='viridis', edgecolor='none', rstride=5, cstride=5)
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('f(x)')
    ax1.set_title('3D 曲面圖')

    #等高線圖
    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contourf(x1, x2, yk, 20, cmap='viridis')
    fig.colorbar(contour, ax=ax2, label='f(x) value')

    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_title('等高線圖')
    ax2.set_aspect('equal')

    # 保存圖形
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"圖表已儲存至 {save_path}")


if __name__ == "__main__":
    functions_to_plot = [ #每個函數的範圍按照題目要求設定
        {
            "func": rosenbrock,
            "range": (-5, 10), 
            "title": "Rosenbrock Function"},
        {
            "func": rastrigin,
            "range": (-5.12, 5.12),
            "title": "Rastrigin Function"},
        {
            "func": griewank,
            "range": (-600, 600),
            "title": "Griewank Function"}
    ]
    
    output_dir = 'results\\plots'
    os.makedirs(output_dir, exist_ok=True)

    # 為每個函數生成圖表
    for item in functions_to_plot:

        plot_function_3d(
            func=item["func"],
            x_range=item["range"],
            y_range=item["range"],
            title=item["title"],
            num_points=1000,
            output_dir=output_dir
        )

    print("所有函數圖表都已生成在 'results\\plots\\' 資料夾中。\n")
