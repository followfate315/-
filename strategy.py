import os
import pandas as pd

def generate_signals():
    # 1. 定位並讀取剛剛存好的 CSV 檔案
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "..", "data", "tsmc_2330_technical.csv")

    if not os.path.exists(data_path):
        print("找不到資料檔，請先執行爬蟲程式！")
        return

#讀取資料，並將 Date 設為索引
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)

#建立一個新欄位叫 'Signal'（訊號），預設為 'Hold' (持有/觀望)
    df['Signal'] = 'Hold'

#用迴圈檢查每一天的均線狀況 (從第 2 天開始檢查，因為需要跟前一天比較)
    for i in range(1, len(df)):
        # 今天的均線
        today_ma5 = df['MA5'].iloc[i]
        today_ma20 = df['MA20'].iloc[i]

#前一天的均線
        yesterday_ma5 = df['MA5'].iloc[i-1]
        yesterday_ma20 = df['MA20'].iloc[i-1]

#排除掉剛開始幾天均線還沒算出來的 NaN 資料
        if pd.isna(today_ma5) or pd.isna(today_ma20) or pd.isna(yesterday_ma5) or pd.isna(yesterday_ma20):
            continue

#判斷黃金交叉：前一天 MA5 <= MA20，但今天 MA5 > MA20
        if yesterday_ma5 <= yesterday_ma20 and today_ma5 > today_ma20:
            df.loc[df.index[i], 'Signal'] = 'Buy (買入)'

#判斷死亡交叉：前一天 MA5 >= MA20，但今天 MA5 < MA20
        elif yesterday_ma5 >= yesterday_ma20 and today_ma5 < today_ma20:
            df.loc[df.index[i], 'Signal'] = 'Sell (賣出/觀望)'

    # 4. 把含有訊號的結果存成新檔案
    output_path = os.path.join(current_dir, "..", "data", "tsmc_signals.csv")
    df.to_csv(output_path)
    print(f"策略分析完成！新檔案已儲存至：{output_path}")

#秀出最新幾天的狀況來瞧瞧
    print("\n--- 最新 10 天的推薦建議預覽 ---")
    print(df[['Close', 'MA5', 'MA20', 'Signal']].tail(10))

if __name__ == "__main__":
    generate_signals()