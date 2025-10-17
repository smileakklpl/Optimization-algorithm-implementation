import os
import pandas as pd

def export_tables_to_excel(csv_path, output_dir):

    # 如果輸出目錄不存在，則建立它
    os.makedirs(output_dir, exist_ok=True)
    print(f"輸出目錄 '{output_dir}' 已準備就緒。")

    df = pd.read_csv(csv_path)

    functions = df['Function'].unique()
    algorithms = df['Algorithm'].unique()

    file_count = 0
    # 遍歷函數和演算法的每種組合
    for func in functions:
        for algo in algorithms:
            # 篩選出當前組合的 DataFrame
            subset = df[(df['Function'] == func) & (df['Algorithm'] == algo)]

            if not subset.empty:
                # 根據維度進行排序
                subset = subset.sort_values(by='Dimension').copy()
                # 將維度D更改為 "D=..." 格式
                subset['Dimension'] = 'D=' + subset['Dimension'].astype(str)
                # 準備資料進行匯出，將 Dimension 設為索引，然後進行轉置
                output_df = subset[['Dimension', 'Best', 'Mean', 'Worst', 'StdDev']].set_index('Dimension').T

                # 定義輸出檔名
                file_name = f"{func}_{algo}.xlsx"
                output_path = os.path.join(output_dir, file_name)

                # 將轉置後的 DataFrame 存檔為 Excel 檔案，並將演算法名稱設為索引標籤 (A1儲存格)
                output_df.to_excel(output_path, index_label=algo)
                print(f"成功建立: {output_path}")
                file_count += 1
    
    print(f"\n總共建立了 {file_count} 個檔案")

def main():

    # 輸入 CSV 檔案的路徑
    csv_file_path = './results/data/summary_statistics.csv'
    
    # 輸出 Excel 檔案的目錄
    output_excel_dir = './results/xlsx_reports'
    
    export_tables_to_excel(csv_file_path, output_excel_dir)

if __name__ == "__main__":
    
    main()
