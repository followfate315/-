import streamlit as st  
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import twstock
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import numpy as np

# 機器學習與深度學習套件（Classifier）
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
# import tensorflow as tf

# 固定隨機種子，確保結果可複現
np.random.seed(42)
# tf.random.set_seed(42)

# 1. 網頁頂級配置
st.set_page_config(page_title="🚀 頂級 AI 量化交易終端系統 - 多模型聯防勝率版", layout="wide")

# ==============================================================================
# 🎨 台股紅漲綠跌（黑金賽博朋克風格）CSS 注入
# ==============================================================================
st.markdown("""
    <style>
    .main { background-color: #0b0f19; }
    h1 { color: #ffffff; font-family: 'Segoe UI', Roboto, sans-serif; font-weight: 850; text-shadow: 0px 0px 20px rgba(88,166,255,0.2); }
    h3 { color: #58a6ff !important; font-weight: 600; letter-spacing: 0.5px; }
    
    .premium-card {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(4px);
        transition: all 0.3s ease;
    }
    .premium-card:hover {
        border-color: #58a6ff;
        box-shadow: 0 8px 32px 0 rgba(88,166,255,0.1);
    }
    
    .decision-panel {
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 0 30px rgba(0,0,0,0.5);
        animation: pulse 2s infinite;
    }
    .decision-buy {
        background: linear-gradient(135deg, rgba(255,123,114,0.15) 0%, rgba(22,27,34,1) 100%);
        border: 2px solid #ff7b72; 
        color: #ff7b72;
        text-shadow: 0 0 15px rgba(255,123,114,0.6);
    }
    .decision-sell {
        background: linear-gradient(135deg, rgba(86,211,100,0.15) 0%, rgba(22,27,34,1) 100%);
        border: 2px solid #56d364; 
        color: #56d364;
        text-shadow: 0 0 15px rgba(86,211,100,0.6);
    }
    .decision-wait {
        background: linear-gradient(135deg, rgba(210,153,34,0.15) 0%, rgba(22,27,34,1) 100%);
        border: 2px solid #d29922; 
        color: #d29922;
        text-shadow: 0 0 15px rgba(210,153,34,0.6);
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 15px rgba(0,0,0,0.5); }
        50% { box-shadow: 0 0 25px rgba(88,166,255,0.15); }
        100% { box-shadow: 0 0 15px rgba(0,0,0,0.5); }
    }
    
    .news-stream-card {
        background: #161b22;
        border-left: 4px solid #ff9b44;
        border-radius: 6px;
        padding: 15px;
        margin-bottom: 12px;
        transition: background-color 0.2s;
    }
    .news-stream-card:hover { background: #1f242c; }
    
    .sentiment-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
        margin-top: 5px;
    }
    .badge-bullish { background-color: rgba(255,123,114,0.2); color: #ff7b72; border: 1px solid #ff7b72; }
    .badge-bearish { background-color: rgba(86,211,100,0.2); color: #56d364; border: 1px solid #56d364; }
    .badge-neutral { background-color: rgba(139,148,158,0.2); color: #8b949e; border: 1px solid #8b949e; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ NEXT-GEN 量化終端：三模型投票勝率與動態風控系統")

# ==============================================================================
# 🛠️ 側邊欄 UX 優化
# ==============================================================================
st.sidebar.markdown("### 🛠️ 量化引擎核心配置")

popular_stocks = {
    "自訂輸入...": "",
    "台積電 (2330)": "2330",
    "鴻海 (2317)": "2317",
    "聯發科 (2454)": "2454",
    "廣達 (2382)": "2382",
    "長榮 (2603)": "2603",
    "緯創 (3231)": "3231",
    "元大台灣50 (0050)": "0050"
}

selected_option = st.sidebar.selectbox("🔥 熱門個股速選：", list(popular_stocks.keys()))

if selected_option == "自訂輸入...":
    input_sid = st.sidebar.text_input("🔍 輸入任意台股代碼 (例如: 2330, 2603):", value="2330")
    stock_id = input_sid.strip()
else:
    stock_id = popular_stocks[selected_option]

stock_name = stock_id
if stock_id in twstock.codes:
    stock_name = f"{twstock.codes[stock_id].name} ({stock_id})"

predict_days = st.sidebar.slider("🔮 AI 預測目標跨度 (交易日)：", min_value=1, max_value=10, value=3, step=1)
target_return_pct = st.sidebar.slider("🎯 定義看漲門檻 (未來N日漲幅%)：", min_value=0.5, max_value=5.0, value=1.5, step=0.5)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 技術指標顯示開關")
show_bollinger = st.sidebar.checkbox("📐 顯示布林通道 (Bollinger Bands)", value=True)
show_kd = st.sidebar.checkbox("📊 顯示 KD 隨機指標 (%K / %D)", value=True)

# ==============================================================================
# 📅 交易日推算工具（自動扣除週末）
# ==============================================================================
def get_future_trade_date(start_date, trade_days):
    current = start_date
    added_days = 0
    while added_days < trade_days:
        current += timedelta(days=1)
        if current.weekday() < 5:
            added_days += 1
    return current

# ==============================================================================
# 💡 台灣市場最小升降單位 (Tick Size) 校正
# ==============================================================================
def align_to_tw_tick_size(price, sid):
    if price is None or np.isnan(price): return None
    if sid.startswith("00"):
        tick = 0.01 if price < 50 else 0.05
    else:
        if price < 10: tick = 0.01
        elif price < 50: tick = 0.05
        elif price < 100: tick = 0.1
        elif price < 500: tick = 0.5
        elif price < 1000: tick = 1.0
        else: tick = 5.0
    return round(price / tick) * tick

# ==============================================================================
# ✨ 技術面多因子特徵工程模組 (擴充量價與波動度)
# ==============================================================================
def build_advanced_features(df):
    df = df.copy()
    
    # 均線與動能偏離
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Daily_Return'] = df['Close'].pct_change() * 100
    df['Bias_MA5'] = (df['Close'] - df['MA5']) / df['MA5']
    df['Bias_MA20'] = (df['Close'] - df['MA20']) / df['MA20']
    
    # 量價變動率與歷史波動度
    df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Vol_ROC'] = df['Volume'].pct_change(5)
    df['Hist_Volatility'] = df['Daily_Return'].rolling(10).std()
    
    # RSI 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI_14'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = macd_line - signal_line

    # 布林通道 (Bollinger Bands)
    std20 = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['MA20'] + (std20 * 2)
    df['BB_Lower'] = df['MA20'] - (std20 * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['MA20']

    # KD 指標 (9, 3, 3)
    low_9 = df['Low'].rolling(9).min()
    high_9 = df['High'].rolling(9).max()
    rsv = ((df['Close'] - low_9) / (high_9 - low_9 + 1e-9)) * 100
    
    k_vals, d_vals = [50.0], [50.0]
    for r in rsv.fillna(50):
        k = (2/3) * k_vals[-1] + (1/3) * r
        d = (2/3) * d_vals[-1] + (1/3) * k
        k_vals.append(k)
        d_vals.append(d)
        
    df['K_9'] = k_vals[1:]
    df['D_9'] = d_vals[1:]

    # ATR 14
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14).mean()

    return df

# ==============================================================================
# ✨ 金融語意情緒分析大腦 (NLP Engine)
# ==============================================================================
def analyze_news_sentiment(title, desc):
    text = (title + desc).lower()
    bullish_words = ['創高', '營收增', '大賺', '買超', '看好', '達標', '利多', '超預期', '調升', '強勁', '擴產', '噴發', '獲利翻倍', '黃金交叉', '追價', '訂單滿', '旺季', '成家', '大漲']
    bearish_words = ['衰退', '重挫', '賣超', '看淡', '保守', '砍單', '利空', '跌破', '調降', '疲弱', '修正', '虧損', '匯損', '死亡交叉', '觀望', '淡季', '縮水', '警訊', '大跌', '慘']
    
    score = 0
    for word in bullish_words:
        if word in text: score += 1
    for word in bearish_words:
        if word in text: score -= 1
        
    if score > 0: return "正面利多", "badge-bullish", score
    elif score < 0: return "負面利空", "badge-bearish", score
    else: return "中立消息", "badge-neutral", score

# ==============================================================================
# 🤖 1. 動態加權三模型聯防大腦 (Weighted Ensemble Classifier Engine)
# ==============================================================================
from sklearn.preprocessing import StandardScaler

@st.cache_resource(ttl=3600, show_spinner=False)
def train_ensemble_ai(df, days, target_pct):
    try:
        df_feat = build_advanced_features(df)
        
        # 1. 建立 Target
        df_feat['Target'] = (df_feat['Close'].shift(-days) / df_feat['Close'] - 1) >= (target_pct / 100.0)
        df_feat['Target'] = df_feat['Target'].astype(int)
        
        # 2. 清除 NaN 與 Inf
        df_clean = df_feat.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(df_clean) < 50:
            return 0.0, "0/3 個模型", {}

        # 3. 準備特徵
        drop_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Target']
        feature_cols = [c for c in df_clean.columns if c not in drop_cols]
        
        X = df_clean[feature_cols]
        y = df_clean['Target']

        # 4. 標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 5. 訓練模型與預測
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X, y)
        rf_prob = rf.predict_proba(X[-1:])[:, 1][0]

        xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, eval_metric='logloss')
        xgb.fit(X, y)
        xgb_prob = xgb.predict_proba(X[-1:])[:, 1][0]

        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
        mlp.fit(X_scaled, y)
        latest_scaled = scaler.transform(X[-1:])
        mlp_prob = mlp.predict_proba(latest_scaled)[:, 1][0]

        # 6. 勝率與投票計算
        models_prob = {
            "Random Forest": rf_prob,
            "XGBoost": xgb_prob,
            "Neural Network (MLP)": mlp_prob
        }
        
        # 統計有多少個模型預測看漲 (機率 >= 0.5)
        bullish_count = sum(1 for p in models_prob.values() if p >= 0.5)
        voting_str = f"{bullish_count}/3 個模型"

        # 加權勝率
        weights = {"Random Forest": 0.30, "XGBoost": 0.45, "Neural Network (MLP)": 0.25}
        ensemble_prob = sum(models_prob[m] * weights[m] for m in models_prob)
        
        # 正確回傳 3 個變數
        return ensemble_prob, voting_str, models_prob

    except Exception as e:
        # 印出錯誤方便調試
        st.error(f"模型計算錯誤：{e}")
        return 0.0, "0/3 個模型", {}
# ==============================================================================
# 2. 【數據源】
# ==============================================================================
@st.cache_data(ttl=1800, show_spinner=False)  
def load_stock_data_local(sid):
    try:
        stock_data = twstock.Stock(sid)
        today = datetime.now()
        start_year = today.year - 1
        raw_data = stock_data.fetch_from(start_year, 1) 
        if not raw_data: return None
        df = pd.DataFrame(raw_data)
        df.rename(columns={'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'capacity': 'Volume'}, inplace=True)
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df = df[df['Close'] > 0].reset_index(drop=True)
    except: return None
    
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Daily_Return'] = df['Close'].pct_change() * 100
    df['prev_MA5'] = df['MA5'].shift(1)
    df['prev_MA20'] = df['MA20'].shift(1)
    golden_cross = (df['prev_MA5'] <= df['prev_MA20']) & (df['MA5'] > df['MA20'])
    death_cross = (df['prev_MA5'] >= df['prev_MA20']) & (df['MA5'] < df['MA20'])
    df['Signal'] = np.select([golden_cross, death_cross], ['Buy (買入)', 'Sell (賣出)'], default='Hold')
    
    df = build_advanced_features(df)
    return df

# ==============================================================================
# 3. 【消息面】
# ==============================================================================
@st.cache_data(ttl=600, show_spinner=False)  
def fetch_stock_news_with_nlp(sid):
    news_list = []
    try:
        url = f"https://tw.stock.yahoo.com/rss?s={sid}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=4)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "xml")
            for item in soup.find_all("item")[:4]:
                title = item.find("title").text if item.find("title") else "無標題"
                link = item.find("link").text if item.find("link") else "#"
                pub_date = item.find("pubDate").text if item.find("pubDate") else ""
                description = item.find("description").text if item.find("description") else "查看內文。"
                
                sentiment_label, badge_class, score = analyze_news_sentiment(title, description)
                
                news_list.append({
                    "title": title, "link": link, "date": pub_date.replace(" +0800", ""), 
                    "desc": description, "sentiment": sentiment_label, "class": badge_class, "score": score
                })
    except: pass
    return news_list

# --- 數據同步加載 ---
with st.spinner(f'🚀 正在載入 {stock_name} 並啟動三模型 (RF/XGB/LSTM) 聯防交叉計算...'):
    df = load_stock_data_local(stock_id)
    news_data = fetch_stock_news_with_nlp(stock_id)

if df is not None:
    df['Date'] = pd.to_datetime(df['Date'])
    ai_results = train_ensemble_ai(df, predict_days, target_return_pct)

    # --- 💎 數據總覽與日期推算 ---
    latest_data = df.iloc[-1]
    price_now = float(latest_data['Close'])
    vol_now_sheets = int(latest_data['Volume'] // 1000)
    
    latest_date_dt = latest_data['Date'].to_pydatetime()
    target_date_dt = get_future_trade_date(latest_date_dt, predict_days)
    target_date_str = target_date_dt.strftime('%Y/%m/%d')
    latest_date_str = latest_date_dt.strftime('%Y/%m/%d')
    
    # 風控數值計算 (ATR & VaR)
    atr_val = latest_data['ATR_14'] if not np.isnan(latest_data['ATR_14']) else 0.0
    var_95 = price_now * (np.percentile(df['Daily_Return'].dropna(), 5) / 100)
    stop_loss_price = align_to_tw_tick_size(price_now - (1.5 * atr_val), stock_id)

    st.markdown(f"### 💎 {stock_name} 實時量化數據總覽 (截至 {latest_date_str})")
    
    col_left, col_right = st.columns([2.5, 1.5])
    
    with col_left:
        sub1, sub2 = st.columns(2)
        with sub1:
            st.markdown(f"""<div class="premium-card">
                <div style="color:#8b949e; font-size:13px;">現下收盤報價 ({latest_date_str})</div>
                <div style="color:#ffffff; font-size:32px; font-weight:800;">${price_now:.1f}</div>
                <div style="color:{'#ff7b72' if latest_data['Daily_Return']>=0 else '#56d364'}; font-size:14px; font-weight:600;">
                    漲跌幅 {latest_data['Daily_Return']:+.2f}%
                </div>
            </div>""", unsafe_allow_html=True)
            st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
            st.markdown(f"""<div class="premium-card">
                <div style="color:#8b949e; font-size:13px;">當日成交量</div>
                <div style="color:#ffffff; font-size:28px; font-weight:700;">{vol_now_sheets:,} <span style="font-size:14px;">張</span></div>
                <div style="color:#8b949e; font-size:12px;">每日股數即時統計</div>
            </div>""", unsafe_allow_html=True)
        with sub2:
            avg_p = ai_results['avg_prob'] * 100 if ai_results else 0
            votes = ai_results['bullish_votes'] if ai_results else 0
            st.markdown(f"""<div class="premium-card" style="border-color:#ff9b44;">
                <div style="color:#ff9b44; font-size:13px;">📅 預測目標：{target_date_str} (>{target_return_pct}%)</div>
                <div style="color:#ff9b44; font-size:32px; font-weight:800;">{avg_p:.1f}% 勝率</div>
                <div style="color:#8b949e; font-size:13px; font-weight:600;">
                    模型看漲投票： <span style="color:#ffffff; font-weight:bold;">{votes} / 3</span> 個模型
                </div>
            </div>""", unsafe_allow_html=True)
            st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
            st.markdown(f"""<div class="premium-card">
                <div style="color:#8b949e; font-size:13px;">均線交叉狀態</div>
                <div style="color:#58a6ff; font-size:20px; font-weight:700;">{latest_data['Signal']}</div>
                <div style="color:#8b949e; font-size:12px;">依據 MA5/MA20 計算</div>
            </div>""", unsafe_allow_html=True)

    with col_right:
        if ai_results:
            avg_p = ai_results['avg_prob']
            votes = ai_results['bullish_votes']
            
            # 使用軟投票機率門檻設定決策條件
            if votes >= 2 and avg_p >= 0.58:
                decision_style = "decision-panel decision-buy"
                decision_title = "🔴 AI 聯防決策：多頭共識強烈"
                decision_body = f"已有 {votes}/3 個 AI 模型達成看漲共識，預估在 <b>{target_date_str}</b> 前達標勝率為 {avg_p*100:.1f}%！建議建立多頭部位，動態停損點設為 ${stop_loss_price:.2f}。"
            elif votes <= 1 or avg_p <= 0.42:
                decision_style = "decision-panel decision-sell"
                decision_title = "🟢 AI 聯防決策：空頭防禦觀望"
                decision_body = f"預估在 <b>{target_date_str}</b> 前達標勝率偏低 ({avg_p*100:.1f}%)。多數 AI 模型顯示動能不足或有修正風險，建議保留現金。"
            else:
                decision_style = "decision-panel decision-wait"
                decision_title = "🟡 AI 聯防決策：多空觀望帶 (信心度適中)"
                decision_body = f"模型間呈現分歧或漲升信心尚待確認（看漲投票: {votes}/3，至 <b>{target_date_str}</b> 加權勝率: {avg_p*100:.1f}%）。建議等待突破 signals。"
        else:
            decision_style = "decision-panel decision-wait"
            decision_title = "⚠️ 數據不足"
            decision_body = "無法計算 AI 模型投票。"

        st.markdown(f"""
            <div class="{decision_style}">
                <div style="font-size:15px; text-transform:uppercase; letter-spacing:1px; margin-bottom:5px;">AI Ensemble Decision</div>
                <div style="font-size:26px; font-weight:800; margin-bottom:10px;">{decision_title}</div>
                <div style="font-size:13px; font-weight:400; line-height:1.6; opacity:0.9;">{decision_body}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='border-bottom: 1px gradient; background: linear-gradient(to right, rgba(88,166,255,0), rgba(88,166,255,0.5), rgba(88,166,255,0)); height: 1px; margin: 25px 0;'></div>", unsafe_allow_html=True)

    # ==============================================================================
    # 🤖 三大 AI 模型獨立勝率診斷板
    # ==============================================================================
    st.markdown(f"### 🤖 三大 AI 模型看漲勝率明細 (預測目標日：{target_date_str})")
    if ai_results:
        mc1, mc2, mc3 = st.columns(3)
        probs = ai_results['models_prob']
        
        with mc1:
            st.markdown(f"""<div class="premium-card">
                <div style="color:#8b949e; font-size:13px;">Random Forest (隨機森林)</div>
                <div style="color:{'#ff7b72' if probs['Random Forest']>=0.5 else '#56d364'}; font-size:26px; font-weight:700;">{probs['Random Forest']*100:.1f}%</div>
                <div style="color:#8b949e; font-size:12px;">適合抗雜訊、盤整型特徵識別</div>
            </div>""", unsafe_allow_html=True)
            
        with mc2:
            st.markdown(f"""<div class="premium-card">
                <div style="color:#8b949e; font-size:13px;">XGBoost (極限梯度提升)</div>
                <div style="color:{'#ff7b72' if probs['XGBoost']>=0.5 else '#56d364'}; font-size:26px; font-weight:700;">{probs['XGBoost']*100:.1f}%</div>
                <div style="color:#8b949e; font-size:12px;">適合極端趨勢與短期非線性特徵</div>
            </div>""", unsafe_allow_html=True)
            
        with mc3:
            st.markdown(f"""<div class="premium-card">
                <div style="color:#8b949e; font-size:13px;">Neural Network (MLP 神經網路)</div>
                <div style="color:{'#ff7b72' if probs['Neural Network (MLP)']>=0.5 else '#56d364'}; font-size:26px; font-weight:700;">{probs['Neural Network (MLP)']*100:.1f}%</div>
                <div style="color:#8b949e; font-size:12px;">適合捕捉非線性特徵與週期規律</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='border-bottom: 1px gradient; background: linear-gradient(to right, rgba(88,166,255,0), rgba(88,166,255,0.5), rgba(88,166,255,0)); height: 1px; margin: 25px 0;'></div>", unsafe_allow_html=True)

    # ==============================================================================
    # 🛡️ 動態風險控管儀表板 (ATR & VaR)
    # ==============================================================================
    st.markdown("### 🛡️ 動態風險控管儀表板 (Risk Management Dashboard)")
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.markdown(f"""<div class="premium-card">
            <div style="color:#8b949e; font-size:13px;">14日真實波動區間 (ATR)</div>
            <div style="color:#58a6ff; font-size:26px; font-weight:700;">±${atr_val:.2f}</div>
            <div style="color:#8b949e; font-size:12px;">每日平均價格劇烈波動度</div>
        </div>""", unsafe_allow_html=True)
    with rc2:
        st.markdown(f"""<div class="premium-card">
            <div style="color:#8b949e; font-size:13px;">建議動態停損價 (1.5x ATR)</div>
            <div style="color:#ff7b72; font-size:26px; font-weight:700;">${stop_loss_price:.2f}</div>
            <div style="color:#8b949e; font-size:12px;">跌破此價位即刻執行防禦離場</div>
        </div>""", unsafe_allow_html=True)
    with rc3:
        st.markdown(f"""<div class="premium-card">
            <div style="color:#8b949e; font-size:13px;">單日 95% 風險價值 (VaR)</div>
            <div style="color:#56d364; font-size:26px; font-weight:700;">${abs(var_95):.2f} / 股</div>
            <div style="color:#8b949e; font-size:12px;">極端行情下的最大預期單日損失</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='border-bottom: 1px gradient; background: linear-gradient(to right, rgba(88,166,255,0), rgba(88,166,255,0.5), rgba(88,166,255,0)); height: 1px; margin: 25px 0;'></div>", unsafe_allow_html=True)

    # ==============================================================================
    # ✨ 圖表區：K 線圖 + 成交量 + 布林通道 + KD 指標圖
    # ==============================================================================
    st.subheader("📉 互動式 K 線、技術指標與成交量軌跡圖")
    
    rows_cnt = 3 if show_kd else 2
    row_h = [0.55, 0.25, 0.2] if show_kd else [0.7, 0.3]
    
    fig = make_subplots(rows=rows_cnt, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_h)
    
    # 1. K線圖
    fig.add_trace(go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="歷史K線",
        increasing_line_color='#ff7b72', increasing_fillcolor='#ff7b72',
        decreasing_line_color='#56d364', decreasing_fillcolor='#56d364'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA5'], mode='lines', name='MA5 均線', line=dict(color='#ff9b44', width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], mode='lines', name='MA20 均線', line=dict(color='#58a6ff', width=1.2)), row=1, col=1)
    
    if show_bollinger:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], mode='lines', name='布林上軌', line=dict(color='rgba(255,255,255,0.35)', width=1, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], mode='lines', name='布林下軌', line=dict(color='rgba(255,255,255,0.35)', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(88,166,255,0.03)'), row=1, col=1)

    # 2. 成交量圖
    vol_colors = ['#ff7b72' if row['Close'] >= row['Open'] else '#56d364' for _, row in df.iterrows()]
    df_vol_sheets = df['Volume'] / 1000

    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df_vol_sheets,
        name="成交量 (張)",
        marker_color=vol_colors
    ), row=2, col=1)

    # 3. KD 指標圖
    if show_kd:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['K_9'], mode='lines', name='K值 (9D)', line=dict(color='#ff7b72', width=1.2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['D_9'], mode='lines', name='D值 (9D)', line=dict(color='#58a6ff', width=1.2)), row=3, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="rgba(255,123,114,0.5)", row=3, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="rgba(86,211,100,0.5)", row=3, col=1)

    fig.update_layout(
        xaxis_rangeslider_visible=False, 
        template="plotly_dark", 
        height=700 if show_kd else 600, 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div style='border-bottom: 1px gradient; background: linear-gradient(to right, rgba(88,166,255,0), rgba(88,166,255,0.5), rgba(88,166,255,0)); height: 1px; margin: 30px 0;'></div>", unsafe_allow_html=True)

    # ==============================================================================
    # ✨ 消息面與 NLP 診斷儀表板
    # ==============================================================================
    st.subheader(f"📰 {stock_name} 最新即時焦點新聞與 AI 情緒診斷")
    
    col_news, col_sentiment_panel = st.columns([2.5, 1.5])
    
    with col_news:
        if news_data:
            pos_count, neg_count, neu_count = 0, 0, 0
            for news in news_data:
                if news['sentiment'] == "正面利多": pos_count += 1
                elif news['sentiment'] == "負面利空": neg_count += 1
                else: neu_count += 1
                
                st.markdown(f"""
                    <div class="news-stream-card">
                        <a style="color:#58a6ff; font-weight:600; text-decoration:none; font-size:15px;" href="{news['link']}" target="_blank">🔗 {news['title']}</a>
                        <div style="color:#8b949e; font-size:11px; margin:4px 0;">⏱️ {news['date']} | 來源：Yahoo 股市 RSS</div>
                        <div style="color:#c9d1d9; font-size:13px; margin-bottom:8px;">{news['desc']}</div>
                        <span class="sentiment-badge {news['class']}">🤖 AI 情緒判定：{news['sentiment']} (權重分數: {news['score']})</span>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("🔎 暫時無法取得該個股的即時焦點新聞。")
            pos_count, neg_count, neu_count = 0, 0, 0
            
    with col_sentiment_panel:
        st.markdown(f"""<div class="premium-card" style="text-align: center; height:100%;">
            <div style="color:#8b949e; font-size:14px; font-weight:bold; margin-bottom:15px;">🧠 即時消息面綜合風向計</div>
            <div style="display: flex; justify-content: space-around; margin-bottom:25px;">
                <div>
                    <div style="font-size:24px;">🔴</div>
                    <div style="color:#ff7b72; font-weight:bold; font-size:18px;">{pos_count} 則</div>
                    <div style="color:#8b949e; font-size:12px;">正面利多</div>
                </div>
                <div>
                    <div style="font-size:24px;">🟢</div>
                    <div style="color:#56d364; font-weight:bold; font-size:18px;">{neg_count} 則</div>
                    <div style="color:#8b949e; font-size:12px;">負面利空</div>
                </div>
                <div>
                    <div style="font-size:24px;">🟡</div>
                    <div style="color:#8b949e; font-weight:bold; font-size:18px;">{neu_count} 則</div>
                    <div style="color:#8b949e; font-size:12px;">中立消息</div>
                </div>
            </div>
            <hr style="border-color:#30363d; margin:15px 0;">
            <div style="font-size:13px; color:#c9d1d9; text-align:left; line-height:1.6;">
                📌 <b>量化大腦解析：</b><br>
                本區塊透過 NLP 語意特徵掃描當前即時市場新聞。結合三模型投票結果與 ATR 停損位，提供全方位的風險與進場防護。
            </div>
        </div>""", unsafe_allow_html=True)
else:
    st.error(f"❌ 找不到股票代碼 [{stock_id}] 的資料，請確認輸入是否為正確的台股代碼。")
