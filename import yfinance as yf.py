import os 
import time
from datetime import datetime
import pandas as pd
import twstock

def get_stock_data(stock_id="2317", start_year=2025, start_month=1, max_retries=3):
    today_str = datetime.now().strftime("%Y-%m-%d")
    print(f"正在從台灣證券交易所 (twstock) 抓取 {stock_id} 歷史資料 (截至今天 {today_str})...")

    # 加入失敗重試機制
    for attempt in range(1, max_retries + 1):
        try:
            time.sleep(2)  # 每次請求前暫停 2 秒，避免觸發 TWSE 頻率限制
            
            if stock_id.startswith("00"):
                stock = twstock.Stock(stock_id, initial_fetch=False)
            else:
                stock = twstock.Stock(stock_id)

            raw_data = stock.fetch_from(start_year, start_month)
            if raw_data:
                break
        except Exception as e:
            print(f"⚠️ 第 {attempt} 次嘗試失敗: {e}")
            if attempt == max_retries:
                print("❌ 已達到最大重試次數，無法取得資料。")
                return None
            time.sleep(3)  # 重試前多等待 3 秒

    try:
        # 轉換成 Pandas DataFrame 並整理欄位名稱
        df = pd.DataFrame(raw_data)
        df.rename(columns={
            'date': 'Date', 'open': 'Open', 'high': 'High', 
            'low': 'Low', 'close': 'Close', 'capacity': 'Volume'
        }, inplace=True)

        # 篩選核心欄位與清洗資料
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df = df[df['Close'] > 0].reset_index(drop=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # 指標計算
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['Daily_Return'] = df['Close'].pct_change() * 100

        return df

    except Exception as e:
        print(f"❌ 資料處理時發生錯誤: {e}")
        return None

if __name__ == "__main__":
    # 使用 input() 讓使用者手動輸入，並自動去除頭尾空格；若直接按 Enter 則預設為 "2317"
    user_input = input("請輸入股票代碼 (預設 2330): ").strip()
    target_stock = user_input if user_input else "2330"

    stock_df = get_stock_data(stock_id=target_stock, start_year=2025, start_month=1)

    if stock_df is not None:
        print("\n--- 抓取成功！最新 10 筆交易日數據預覽 ---")
        print(stock_df.tail(10))

        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, "..", "data")
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        save_path = os.path.join(data_dir, f"stock_{target_stock}_technical.csv")
        stock_df.to_csv(save_path)
        print(f"\n✅ 資料已成功更新並儲存至: {save_path}")