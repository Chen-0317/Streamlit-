import yfinance as yf
import pandas as pd
import requests
import time
from bs4 import BeautifulSoup
import streamlit as st
from datetime import datetime, timedelta
import platform
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
import plotly.express as px
import os

# ==================== 系統字體設定 ====================
if platform.system() == 'Windows':
    font_family = 'Microsoft JhengHei'
elif platform.system() == 'Darwin':
    font_family = 'Heiti TC'
else:
    font_family = 'Noto Sans CJK TC'

# ==================== 匯率資料 ====================
currency_code_map = {
    'USD': '美金', 'EUR': '歐元', 'JPY': '日圓', 'CNY': '人民幣',
    'GBP': '英鎊', 'AUD': '澳元', 'CAD': '加拿大元', 'CHF': '瑞士法郎',
    'HKD': '港幣', 'SGD': '新加坡元', 'TWD': '台幣'
}

def get_exchange_rates():
    url = f"https://rate.bot.com.tw/xrt?Lang=zh-TW"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        currency_cells = soup.find_all('td', {'data-table': '幣別'})
        data = []
        for cell in currency_cells:
            name = cell.get_text(strip=True).lower()
            buy = cell.find_next('td', {'data-table': '本行即期買入'})
            sell = cell.find_next('td', {'data-table': '本行即期賣出'})
            data.append({
                "幣別": name.split('(')[0].strip(),
                "即期買入": buy.get_text(strip=True) if buy else "N/A",
                "即期賣出": sell.get_text(strip=True) if sell else "N/A",
                "日期": datetime.now().strftime('%Y-%m-%d')
            })
        return pd.DataFrame(data)
    else:
        st.error("無法查詢匯率資料")
        return pd.DataFrame()

def get_exchange_rate_data(pairs):
    if isinstance(pairs, list):
        pairs = [pair.upper() for pair in pairs]
    df = yf.download(pairs, period="30d", interval="1d")
    if df.empty:
        st.warning("無法獲取匯率數據")
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close'].reset_index().melt(id_vars="Date", var_name="貨幣對", value_name="匯率")
    else:
        df = df[['Close']].rename(columns={'Close': '匯率'})
        df = df.reset_index()
        df['貨幣對'] = pairs[0] if isinstance(pairs, list) and len(pairs) == 1 else '未知'
    df.rename(columns={'Date': '日期'}, inplace=True)
    df['日期'] = pd.to_datetime(df['日期']).dt.tz_localize(None)
    return df

def plot_exchange_rate(df):
    fig = px.line(df, x='日期', y='匯率', color='貨幣對', title='匯率趨勢')
    fig.update_layout(font_family=font_family)
    st.plotly_chart(fig, use_container_width=True)

# ==================== 股票資料 ====================
def get_valid_tickers(tickers):
    valid = []
    for t in tickers:
        try:
            if not yf.Ticker(t).history(period="30d").empty:
                valid.append(t)
        except:
            pass
    return valid

def get_date_range():
    mode = st.sidebar.selectbox('選擇時間:', ['固定範圍', '自訂範圍'])
    today = datetime.today()
    if mode == '固定範圍':
        days = st.sidebar.selectbox('時間區間:', ['1M', '3M', '6M', '1Y'])
        delta = {'1M': 30, '3M': 90, '6M': 180, '1Y': 365}[days]
        return today - timedelta(days=delta), today
    else:
        start = st.sidebar.date_input('起始日期', today - timedelta(days=30))
        end = st.sidebar.date_input('結束日期', today)
        return start, end

def calculate_technical_indicators(df, rsi_period):
    result = {}

    if isinstance(df.columns, pd.MultiIndex):
        # 判斷哪個 level 是 ticker（通常是股票代碼含 ".TW"）
        level_values = df.columns.get_level_values(0)
        is_ticker_first = all(['.TW' in str(v) or v.isdigit() for v in level_values[:5]])
        ticker_level = 0 if is_ticker_first else 1
        field_level = 1 - ticker_level

        for ticker in df.columns.levels[ticker_level]:
            try:
                sub_df = df.xs(ticker, axis=1, level=ticker_level).copy()
                if 'Close' not in sub_df.columns:
                    st.warning(f"{ticker} 缺少 'Close' 欄位，跳過此股票")
                    continue

                sub_df['MA20'] = sub_df['Close'].rolling(window=20).mean()
                std = sub_df['Close'].rolling(window=20).std()
                sub_df['UpperBand'] = sub_df['MA20'] + 2 * std
                sub_df['LowerBand'] = sub_df['MA20'] - 2 * std

                delta = sub_df['Close'].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.rolling(window=rsi_period).mean()
                avg_loss = loss.rolling(window=rsi_period).mean()
                rs = avg_gain / avg_loss
                sub_df['RSI'] = 100 - (100 / (1 + rs))

                result[ticker] = sub_df
            except Exception as e:
                st.warning(f"{ticker} 計算失敗：{e}")
    else:
        if 'Close' not in df.columns:
            st.warning("資料中沒有 'Close' 欄位，無法計算技術指標")
            return {}

        df = df.copy()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['UpperBand'] = df['MA20'] + 2 * std
        df['LowerBand'] = df['MA20'] - 2 * std

        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        result["SINGLE"] = df

    return result


def plot_stock(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'))
    if 'MA20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20'))
        fig.add_trace(go.Scatter(x=df.index, y=df['UpperBand'], name='Upper Band', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['LowerBand'], name='Lower Band', line=dict(dash='dot')))
    fig.update_layout(title=f'{ticker} 股票走勢', font_family=font_family)
    st.plotly_chart(fig, use_container_width=True)

    if 'RSI' in df.columns:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
        fig_rsi.update_layout(title=f'{ticker} RSI', yaxis_range=[0, 100], font_family=font_family)
        st.plotly_chart(fig_rsi, use_container_width=True)

# ==================== 取得台灣股票代碼列表 ====================
@st.cache_data
def get_twse_stock_list_from_local(filename="twse_stock_list_split.csv"):
    try:
        # 讀取 CSV 文件並回傳
        df = pd.read_csv(filename, encoding='utf-8')
        
        # 檢查是否有正確的欄位
        if 'code' not in df.columns or 'name' not in df.columns:
            st.warning("CSV 文件缺少必要的欄位")
            return pd.DataFrame(columns=['code', 'name', 'display'])
        
        # 如果有必要可以進行欄位清理
        df['display'] = df['code'] + ' ' + df['name']
        return df
    except Exception as e:
        st.error(f"載入股票清單失敗：{e}")
        return pd.DataFrame(columns=['code', 'name', 'display'])

# ==================== 股票查詢功能 ====================
def stock_query():
    st.sidebar.subheader("股票查詢")
    stock_list_df = get_twse_stock_list_from_local()
    if stock_list_df.empty:
        st.warning("無法載入股票代碼列表，請稍後再試。")
        return

    search_input = st.sidebar.text_input("輸入股票代碼或前幾位代碼（例如：2330, 0050 或 00, 13）：").strip()

    if search_input:
        if len(search_input) == 2 and search_input.isdigit():  # 若輸入兩位數字，顯示符合的選項
            matched_stocks = stock_list_df[stock_list_df['code'].str.startswith(search_input)]
            if not matched_stocks.empty:
                # 提供下拉選單供用戶選擇股票代碼
                selected_stock = st.selectbox("選擇股票代碼", matched_stocks['display'])
                st.write(f"你選擇了股票：{selected_stock}")
                
                # 顯示選中的股票的技術指標分析圖
                ticker = selected_stock.split(' ')[0]  # 獲取股票代碼部分
                valid_tickers = get_valid_tickers([f"{ticker}.TW"])
                if valid_tickers:
                    rsi_period = st.sidebar.number_input("RSI 週期", 5, 30, 14)
                    start, end = get_date_range()
                    data = yf.download(valid_tickers, start=start, end=end, group_by="ticker", threads=True)
                    df_result = calculate_technical_indicators(data, rsi_period)
                    
                    if isinstance(df_result, dict):
                        for code, df in df_result.items():
                            st.subheader(f"\U0001F4C8 {code} 技術分析圖")
                            plot_stock(df, code)
                    else:
                        st.subheader(f"\U0001F4C8 {valid_tickers[0]} 技術分析圖")
                        plot_stock(df_result, valid_tickers[0])
            else:
                st.warning(f"沒有找到以 '{search_input}' 開頭的股票代碼")
        else:
            # 若用戶輸入股票代碼（單支或多支）
            codes = [code.strip() for code in search_input.split(',') if code.strip()]
            tickers = [f"{code}.TW" for code in codes]

            valid_tickers = get_valid_tickers(tickers)
            if not valid_tickers:
                st.warning("找不到有效的股票代碼")
                return

            rsi_period = st.sidebar.number_input("RSI 週期", 5, 30, 14)
            start, end = get_date_range()

            # 多支股票一次下載
            data = yf.download(valid_tickers, start=start, end=end, group_by="ticker", threads=True)

            df_result = calculate_technical_indicators(data, rsi_period)

            if isinstance(df_result, dict):
                for code, df in df_result.items():
                    st.subheader(f"\U0001F4C8 {code} 技術分析圖")
                    plot_stock(df, code)
            else:
                # 只查一支的時候 fallback
                st.subheader(f"\U0001F4C8 {valid_tickers[0]} 技術分析圖")
                plot_stock(df_result, valid_tickers[0])
    else:
        st.write("請在上方輸入股票代碼或名稱的前幾位以進行查詢。")


# ==================== 主程式 ====================
def main():
    st.set_page_config(layout="wide")
    st.title("\U0001F4C8 匯率與股票視覺化儀表板")
    menu = st.sidebar.radio("功能選單", ["匯率查詢", "股票查詢"])

    if menu == "匯率查詢":
        pairs_input = st.sidebar.text_input("輸入匯率代碼 (例: USDTWD=X, EURTWD=X)")
        if pairs_input:
            pairs = [p.strip().upper() for p in pairs_input.split(',') if p.strip()]
            df = get_exchange_rate_data(pairs)
            if not df.empty:
                st.dataframe(df)
                plot_exchange_rate(df)
        else:
            st_autorefresh(interval=300000, key="refresh")
            df = get_exchange_rates()
            if not df.empty:
                st.dataframe(df)

    elif menu == "股票查詢":
        stock_query()

if __name__ == '__main__':
    main()