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
import numpy as np
import talib as ta

# ==================== 系統字體設定 ====================
if platform.system() == 'Windows':
    font_family = 'Microsoft JhengHei'
elif platform.system() == 'Darwin':
    font_family = 'Heiti TC'
else:
    font_family = 'Noto Sans CJK TC'

# ==================== 匯率資料 ====================
#currency_code_map = {
#    'USD': '美金', 'EUR': '歐元', 'JPY': '日圓', 'CNY': '人民幣',
#   'GBP': '英鎊', 'AUD': '澳元', 'CAD': '加拿大元', 'CHF': '瑞士法郎',
#    'HKD': '港幣', 'SGD': '新加坡幣', 'TWD': '台幣', 'ZAR': '南非幣',
#    'ZND': '紐西蘭幣', 'SEK': '瑞典幣', 'MXN': '墨西哥披索', 'THB': '泰銖'
#}
currency_options = [
    'USD 美金', 'EUR 歐元', 'JPY 日圓', 'CNY 人民幣',
    'GBP 英鎊', 'AUD 澳元', 'CAD 加拿大元', 'CHF 瑞士法郎',
    'HKD 港幣', 'SGD 新加坡幣', 'TWD 台幣', 'ZAR 南非幣',
    'ZND 紐西蘭幣', 'SEK 瑞典幣', 'MXN 墨西哥披索', 'THB 泰銖'
]
# 初始化 Session State 並檢查有效性
if "from_currency" not in st.session_state or st.session_state["from_currency"] not in currency_options:
    st.session_state["from_currency"] = currency_options[0]

if "to_currency" not in st.session_state or st.session_state["to_currency"] not in currency_options:
    st.session_state["to_currency"] = currency_options[0]


def get_exchange_rate(pair):
    df = yf.download(pair, period="30d", interval="1d", auto_adjust=False)
    
    if df.empty:
        st.warning("無法取得匯率資料，請確認貨幣代碼是否正確")
        return None, None

    # 檢查是否為 MultiIndex 欄位
    if isinstance(df.columns, pd.MultiIndex):
        # 取得正確的 close 資料（多層欄位）
        try:
            close_series = df['Close'][pair]  # 例如 df['Close']['USDUSD=X']
        except KeyError:
            st.warning("匯率欄位找不到，請確認匯率代碼是否正確")
            return None, None
    else:
        try:
            close_series = df['Close']
        except KeyError:
            st.warning("匯率欄位缺失")
            return None, None

    # 建立乾淨 dataframe
    close_df = close_series.dropna().reset_index()
    close_df.rename(columns={'Date': '日期', pair: '匯率'}, inplace=True)
    close_df['日期'] = pd.to_datetime(close_df['日期']).dt.tz_localize(None)

    # 解析最後一筆匯率
    try:
        value = close_df['匯率'].iloc[-1]
        rate = float(np.array(value).flatten()[0]) if isinstance(value, (np.ndarray, list)) else float(value)
        if pd.isna(rate):
            raise ValueError("匯率為 NaN")
    except Exception as e:
        st.error(f"無法解析最後一筆匯率資料：{e}")
        return None, close_df

    return rate, close_df

def get_exchange_rate_data(pairs):
    if isinstance(pairs, list):
        pairs = [pair.upper() for pair in pairs]
    df = yf.download(pairs, period="30d", interval="1d", auto_adjust=False)
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

def plot_exchange_rate(df, from_currency, to_currency):
    fig = go.Figure()

    # 使用 '匯率' 作為 y 軸，'日期' 作為 x 軸
    fig.add_trace(go.Scatter(x=df['日期'], y=df['匯率'], mode='lines', name=f'{from_currency}/{to_currency} 匯率'))
    
    fig.update_layout(
        title=f"{from_currency}/{to_currency} 匯率走勢圖",
        xaxis_title="日期",
        yaxis_title="匯率",
        hovermode="x unified",  # 鼠標滑過顯示所有數據點
        dragmode="zoom"  # 啟用放大縮小功能
    )
    
    return fig

# 定義對調幣別的 callback
def swap_currencies():
    from_currency = st.session_state["from_currency"]
    to_currency = st.session_state["to_currency"]
    st.session_state["from_currency"] = to_currency
    st.session_state["to_currency"] = from_currency

def exchange_rate_app():
    st.subheader("💱 匯率換算與趨勢圖")

    # 初始化 Session State
    if "from_currency" not in st.session_state:
        st.session_state["from_currency"] = ""
    if "to_currency" not in st.session_state:
        st.session_state["to_currency"] = ""

    col1, col2, col3 = st.columns([3, 1.5, 3])

    with col1:
        from_currency = st.selectbox("原幣別", currency_options, key="from_currency")
        amount = st.number_input("金額", min_value=0.0, value=100.0, step=10.0)

    with col2:
        st.button("🔁 幣別對調", on_click=swap_currencies, use_container_width=True)

    with col3:
        to_currency = st.selectbox("欲換成幣別", currency_options, key="to_currency")

    # 匯率與圖表顯示
    if st.session_state["from_currency"] and st.session_state["to_currency"]:
        from_currency_code = st.session_state["from_currency"][:3]
        to_currency_code = st.session_state["to_currency"][:3]
        query_code = f"{from_currency_code}{to_currency_code}=X"

        rate, df = get_exchange_rate(query_code)
        if rate is not None:
            try:
                rate = float(rate)
                converted = amount * rate
                st.metric(label="可兌換金額", value=f"{converted:,.2f} {st.session_state['to_currency']}", delta=f"匯率：{rate:.4f}")
                plot_exchange_rate(df, st.session_state["from_currency"], st.session_state["to_currency"])
            except Exception as e:
                st.error(f"匯率顯示錯誤：{e}")
        else:
            st.warning("無法取得匯率資料，請確認幣別組合是否正確。")

    # 顯示匯率走勢圖
    if df is not None:
        fig = plot_exchange_rate(df, from_currency, to_currency)
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
        
        # 加上 RSI 區間線
        fig_rsi.add_hline(y=70, line=dict(color='red', dash='dash'), annotation_text='超買', annotation_position='top left')
        fig_rsi.add_hline(y=30, line=dict(color='green', dash='dash'), annotation_text='超賣', annotation_position='bottom left')
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
        
        # ▼ 顯示當日大盤指數收盤價
        st.markdown("---")
        st.subheader("📊 當日大盤指數")

        try:
            # 多抓一天避免遇到休市日
            dow = yf.download("^DJI", period="2d", interval="1d", group_by="ticker")
            twii = yf.download("^TWII", period="2d", interval="1d", group_by="ticker")
        
            # 重設欄位名稱，將多層欄位扁平化
            dow.columns = [col[1] if isinstance(col, tuple) else col for col in dow.columns]
            twii.columns = [col[1] if isinstance(col, tuple) else col for col in twii.columns]
            
            if not dow.empty and not twii.empty:
                dow_close = dow['Close'].dropna().iloc[-1] if 'Close' in dow.columns else None
                twii_close = twii['Close'].dropna().iloc[-1] if 'Close' in twii.columns else None
                
                if dow_close is not None and twii_close is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("道瓊指數（^DJI）", f"{dow_close:,.2f}")
                    with col2:
                        st.metric("台灣加權指數（^TWII）", f"{twii_close:,.2f}")
                else:
                    st.warning("大盤指數的收盤價資料有問題")
            else:
                st.warning("無法獲取大盤指數資料")
            
            # 顯示今年以來的道瓊與台灣加權指數折線圖
            ytd_start = f"{pd.Timestamp.today().year}-01-01"
            
            # 下載資料
            dow = yf.download("^DJI", start=ytd_start, interval="1d", group_by='ticker')
            twii = yf.download("^TWII", start=ytd_start, interval="1d", group_by='ticker')
            
            # 檢查資料是否有效
            if not dow.empty and not twii.empty:
                # 重設欄位名稱並扁平化
                dow.columns = [col[1] if isinstance(col, tuple) else col for col in dow.columns]
                twii.columns = [col[1] if isinstance(col, tuple) else col for col in twii.columns]
                
                dow = dow.reset_index()
                dow['Index'] = '道瓊 (^DJI)'
                
                twii = twii.reset_index()
                twii['Index'] = '台灣加權 (^TWII)'
                
                df_combined = pd.concat([
                    dow[['Date', 'Close', 'Index']],
                    twii[['Date', 'Close', 'Index']]
                ])
                
                fig = px.line(
                    df_combined,
                    x='Date',
                    y='Close',
                    color='Index',
                    title="2025 年至今：道瓊與台灣加權指數走勢",
                    markers=True
                )
                fig.update_layout(yaxis_tickformat=",", height=450)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("指數資料不足，無法繪製走勢圖")

        except Exception as e:
            st.error(f"讀取大盤指數時發生錯誤：{e}")

        if len(search_input) == 2 and search_input.isdigit():
            matched_stocks = stock_list_df[stock_list_df['code'].str.startswith(search_input)]
            if not matched_stocks.empty:
                selected_stock = st.selectbox("選擇股票代碼", matched_stocks['display'])
                st.write(f"你選擇了股票：{selected_stock}")
                ticker = selected_stock.split(' ')[0]
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
            codes = [code.strip() for code in search_input.split(',') if code.strip()]
            tickers = [f"{code}.TW" for code in codes]
            valid_tickers = get_valid_tickers(tickers)
            if not valid_tickers:
                st.warning("找不到有效的股票代碼")
                return

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
        st.write("請輸入股票代碼或名稱的前幾位以進行查詢。")

# ==================== 主程式 ====================
def main():
    st.set_page_config(layout="wide")
    st.title("📈 匯率與股票視覺化儀表板")
    menu = st.sidebar.radio("功能選單", ["匯率查詢", "股票查詢"])

    if menu == "匯率查詢":
        exchange_rate_app()

    elif menu == "股票查詢":
        stock_query()

if __name__ == '__main__':
    main()