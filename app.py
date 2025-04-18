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
        pairs = ','.join(pairs)
    df = yf.download(pairs, period="30d", interval="1d")
    if df.empty:
        st.warning("無法獲取匯率數據")
        return pd.DataFrame()
    df = df['Close'].reset_index().melt(id_vars="Date", var_name="貨幣對", value_name="匯率")
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
    if isinstance(df.columns, pd.MultiIndex):
        result = {}
        for ticker in df.columns.levels[0]:
            data = df[ticker].copy()
            data['MA20'] = data['Close'].rolling(window=20).mean()
            data['UpperBand'] = data['MA20'] + 2 * data['Close'].rolling(window=20).std()
            data['LowerBand'] = data['MA20'] - 2 * data['Close'].rolling(window=20).std()
            delta = data['Close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=rsi_period).mean()
            avg_loss = loss.rolling(window=rsi_period).mean()
            rs = avg_gain / avg_loss
            data['RSI'] = 100 - (100 / (1 + rs))
            result[ticker] = data
        return result
    else:
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['UpperBand'] = df['MA20'] + 2 * df['Close'].rolling(window=20).std()
        df['LowerBand'] = df['MA20'] - 2 * df['Close'].rolling(window=20).std()
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df

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

# ==================== 取得台灣股票代碼列表（使用 CSV 快取 ==================== 
@st.cache_data
def get_twse_stock_list(use_cache=True, cache_filename="twse_stock_list.csv", max_age_days=1):
    import re
    def is_cache_valid(file_path):
        if not os.path.exists(file_path):
            return False
        modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        return (datetime.now() - modified_time).days < max_age_days

    if use_cache and is_cache_valid(cache_filename):
        try:
            return pd.read_csv(cache_filename, encoding='utf-8')
        except Exception as e:
            st.warning(f"載入本地快取失敗：{e}，改為重新下載...")

    try:
        url = 'https://isin.twse.com.tw/isin/class_i.jsp?kind=1'
        headers = {'User-Agent': 'Mozilla/5.0'}
        tables = pd.read_html(url, encoding='utf-8')

        df = None
        for table in tables:
            if table.shape[1] >= 5 and table.columns[0] == '有價證券代號及名稱':
                df = table.copy()
                break

        if df is None:
            raise ValueError("找不到包含『有價證券代號及名稱』欄位的表格")

        df.columns = df.iloc[0]  # 第一列作為欄位名稱
        df = df[1:]  # 去除標題列
        df = df[['有價證券代號及名稱']].dropna()

        # 解析代號與名稱
        df[['code', 'name']] = df['有價證券代號及名稱'].str.extract(r'(\d+)\s+(.+)')
        df = df[df['code'].notna()]
        df['display'] = df['code'] + ' ' + df['name']
        df = df[['code', 'name', 'display']]

        df.to_csv(cache_filename, index=False, encoding='utf-8')
        return df
    except Exception as e:
        st.error(f"無法取得台灣股票代碼列表：{e}")
        return pd.DataFrame(columns=['code', 'name', 'display'])

# ==================== 股票查詢功能 ====================
def stock_query():
    st.sidebar.subheader("股票查詢")
    stock_list_df = get_twse_stock_list()
    if stock_list_df.empty:
        st.warning("無法載入股票代碼列表，請稍後再試。")
        return

    search_input = st.sidebar.text_input("輸入股票代碼或名稱的前幾位：").strip()
    if search_input:
        filtered_df = stock_list_df[stock_list_df['code'].str.startswith(search_input) | stock_list_df['name'].str.contains(search_input)]
        options = filtered_df['display'].tolist()
        if options:
            selected = st.sidebar.selectbox("選擇股票：", options)
            selected_code = selected.split(' ')[0]
            ticker = f"{selected_code}.TW"
            st.write(f"您選擇的股票代碼為：{ticker}")
            rsi_period = st.sidebar.number_input("RSI 週期", 5, 30, 14)
            start, end = get_date_range()
            data = yf.download(ticker, start=start, end=end)
            if not data.empty:
                df = calculate_technical_indicators(data, rsi_period)
                st.subheader("📊 股票技術分析圖")
                plot_stock(df, ticker)
            else:
                st.warning("找不到此股票的歷史資料")
        else:
            st.warning("未找到符合條件的股票，請嘗試其他輸入。")
    else:
        st.write("請在上方輸入股票代碼或名稱的前幾位以進行查詢。")

# ==================== 主程式 ====================
def main():
    st.set_page_config(layout="wide")
    st.title("📈 匯率與股票視覺化儀表板")
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
