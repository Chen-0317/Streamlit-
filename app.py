import yfinance as yf
import pandas as pd
import requests
import time
from bs4 import BeautifulSoup
import streamlit as st
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import platform

# 設定字型的 URL
font_url = "https://github.com/Chen-0317/fonts/blob/main/NotoSansTC-Regular.ttf"

# 使用 st.markdown 加載字型
st.markdown(f"""
    <style>
    @font-face {{
        font-family: 'Noto Sans CJK TC';
        src: url('{font_url}');
    }}
    body {{
        font-family: 'Noto Sans CJK TC', sans-serif;
    }}
    </style>
""", unsafe_allow_html=True)

# 自動判斷系統並設定中文字型
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Microsoft JhengHei'
elif platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'Heiti TC'
else:  # Linux (例如 Google Colab)
    plt.rcParams['font.family'] = 'Noto Sans CJK TC'


plt.rcParams['axes.unicode_minus'] = False    # 解決負號 '-' 顯示問題

# 英文幣別代碼與中文名稱對應的字典
currency_code_map = {
    'USD': '美金',
    'EUR': '歐元',
    'JPY': '日圓',
    'CNY': '人民幣',
    'GBP': '英鎊',
    'AUD': '澳元',
    'CAD': '加拿大元',
    'CHF': '瑞士法郎',
    'HKD': '港幣',
    'SGD': '新加坡元',
    'TWD': '台幣'
    # 可以在這裡擴展更多的幣別對應
}

# 匯率查詢函數
def get_exchange_rates(currency_input=None):
    url = f"https://rate.bot.com.tw/xrt?Lang=zh-TW"
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        currency_cells = soup.find_all('td', {'data-table': '幣別'})
        
        data = []
        for currency_cell in currency_cells:
            currency_name = currency_cell.get_text(strip=True).lower()
            buy_price = currency_cell.find_next('td', {'data-table': '本行即期買入'})
            sell_price = currency_cell.find_next('td', {'data-table': '本行即期賣出'})
            cash_buy_price = currency_cell.find_next('td', {'data-table': '本行現金買入'})
            cash_sell_price = currency_cell.find_next('td', {'data-table': '本行現金賣出'})
            
            buy_price_value = buy_price.get_text(strip=True) if buy_price else "N/A"
            sell_price_value = sell_price.get_text(strip=True) if sell_price else "N/A"
            cash_buy_price_value = cash_buy_price.get_text(strip=True) if cash_buy_price else "N/A"
            cash_sell_price_value = cash_sell_price.get_text(strip=True) if cash_sell_price else "N/A"
            
            clean_currency_name = currency_name.split('(')[0].strip()
            data.append({
                "幣別": clean_currency_name,
                "即期買入": buy_price_value,
                "即期賣出": sell_price_value,
                "現金買入": cash_buy_price_value,
                "現金賣出": cash_sell_price_value,
                "日期": datetime.now().strftime('%Y-%m-%d')  # 保存日期
            })
        
        df = pd.DataFrame(data)   
        return df
    else:
        st.error(f"無法查詢匯率資料，HTTP狀態碼: {response.status_code}")
        return pd.DataFrame()


# 匯率查詢函數，使用 yfinance 下載匯率數據
def get_exchange_rate_data(currencies):
    # 處理 currency_input，確保輸入格式正確並轉換成大寫
    if isinstance(currencies, list):
        currencies = ','.join(currencies)  # 將列表轉為字符串
    
    currencies = currencies.upper()  # 確保是大寫
    
    # 設置抓取的資料源 URL，這裡需要考慮 Yahoo Finance 數據的格式
    # 確保匯率符號的格式是正確的，像是 "USDNTD=X" 是正確的
    data = yf.download(currencies, period="30d", interval="1d")  # 下載過去 30 天的資料

    # 檢查是否有資料
    if data.empty:
        st.warning("無法獲取匯率數據，請檢查貨幣對格式是否正確。")
        return pd.DataFrame()

    # 只保留收盤價
    data = data['Close']
    data.reset_index(inplace=True)
    data = data.melt(id_vars=["Date"], var_name="貨幣對", value_name="匯率")
    data.rename(columns={'Date': '日期'}, inplace=True)  # 這裡確保列名為 "日期"

    # 轉換 '日期' 列為 datetime 類型，並移除時區資訊
    data['日期'] = pd.to_datetime(data['日期']).dt.tz_localize(None)
    
    # 設置時間範圍：最近 1 個月，每12小時一個數據點
    today = datetime.today()
    start_date = today - timedelta(days=30)
    
    # 篩選最近 30 天的資料
    df_filtered = data[(data['日期'] >= start_date) & (data['日期'] <= today)]

    # 返回處理過後的數據
    return df_filtered


# 顯示匯率折線圖
def plot_exchange_rate_trend(df, currencies):
    # 濾除無效資料
    df['匯率'] = pd.to_numeric(df['匯率'], errors='coerce')

    # 設置時間範圍：最近 1 個月，每小時一個數據點
    today = datetime.today()
    start_date = today - timedelta(days=30)
    
    # 處理日期並設置為時間索引
    df['日期'] = pd.to_datetime(df['日期'])
    df_filtered = df[(df['日期'] >= start_date) & (df['日期'] <= today)]

    # 若沒有找到數據，返回提示
    if df_filtered.empty:
        st.warning("沒有找到匯率數據，請稍後再試")
        return

    # 從貨幣對中提取出貨幣名稱
    currency_pair = currencies[0].replace("=X", "")  # 去除 '=X'
    base_currency, target_currency = currency_pair[:3], currency_pair[3:]  # 提取前3個字符為基礎貨幣，後3個字符為目標貨幣
    
    
    # 轉換為中文貨幣名稱
    base_currency_name = currency_code_map.get(base_currency, base_currency)  # 如果找不到對應中文名，則使用原代碼
    target_currency_name = currency_code_map.get(target_currency, target_currency)
    
    # 動態設置標題
    title = f"{base_currency_name} 兌 {target_currency_name} 過去30天匯率走勢"
    
    # 畫圖
    plt.figure(figsize=(10, 6))
    
    # 畫每個貨幣對的匯率折線圖
    for currency in df_filtered['貨幣對'].unique():
        data_filtered = df_filtered[df_filtered['貨幣對'] == currency]
        sns.lineplot(data=data_filtered, x="日期", y="匯率", label=currency, marker="o")
    
    # 顯示幣別名稱
    plt.title(title, fontsize=14)

    # 設定X軸、Y軸的標籤和格式
    plt.xlabel("日期", fontsize=12)
    plt.ylabel("匯率", fontsize=12)
    plt.xticks(rotation=45)  # 讓日期顯示更清晰
    plt.grid()  # 格線
    plt.legend()

    # 顯示圖表
    st.pyplot(plt)

# 股票數據查詢函數
def get_valid_tickers(tickers):
    valid_tickers = []
    invalid_tickers = []

    # 檢查是否有有效的股票代碼
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="30d", interval="1h")  # 下載過去 30 天的資料
            if not data.empty:
                valid_tickers.append(ticker)
            else:
                invalid_tickers.append(ticker)
        except Exception as e:
            invalid_tickers.append(ticker)

    if invalid_tickers:
        st.warning(f"以下股票代碼無效: {', '.join(invalid_tickers)}")

    return valid_tickers


def plot_stock_trend(df, ticker):
    st.line_chart(df['Close'])
            
def get_date_range():
    while True:
        mode = st.sidebar.selectbox('時間 : 1. 固定範圍 (1M/3M/6M/1Y) 2. 自定義範圍: ', ['1', '2'])
        if mode not in ['1','2']:
            st.warning('輸入無效，請重新選擇')
            continue
            
        today = datetime.today()

        if mode == '1':
            time_range = st.sidebar.selectbox('選擇時間範圍(1M/3M/6M/1Y): ', ['1M', '3M', '6M', '1Y'])
            range_map = {'1M': 30, '3M': 90, '6M': 180, '1Y': 365}
            start_date = today - timedelta(days=range_map[time_range])
            return start_date.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d')

        elif mode == '2':  # 自定義範圍
            # 不進行處理，跳過直到用戶輸入結束日期
            st.warning('請在左側輸入自定義日期範圍。')

            start_date_str = st.sidebar.text_input("請輸入起始日期 (格式: YYYY-MM-DD): ").strip()
            if not start_date_str:
                continue  # 如果未輸入起始日期，跳過並繼續等待輸入
            
            try:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
                if start_date > today:
                    st.warning('起始日期不能在未來，請輸入正確的日期')
                    continue  # 重新進入循環，直到用戶輸入正確日期
            except ValueError:
                st.warning('格式錯誤，請輸入正確的日期格式 (YYYY-MM-DD)')
                continue  # 日期格式錯誤時繼續讓用戶輸入
            
            # 這裡開始輸入結束日期
            end_date_str = st.sidebar.text_input("請輸入結束日期 (格式: YYYY-MM-DD): ").strip()
            if not end_date_str:
                continue  # 如果未輸入結束日期，跳過並繼續等待輸入
            
            try:
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
                if end_date > today:
                    st.warning('結束日期不能在未來，請輸入正確的日期')
                    continue  # 重新進入循環，直到用戶輸入正確日期
                elif end_date < start_date:
                    st.warning('結束日期不能早於起始日期，請重新輸入')
                    continue  # 重新進入循環，直到用戶輸入正確日期
                else:
                    # 如果結束日期是有效的，返回範圍
                    return start_date_str, end_date_str

            except ValueError:
                st.warning('格式錯誤，請輸入正確的結束日期格式 (YYYY-MM-DD)')
                continue  # 日期格式錯誤時繼續讓用戶輸入

# 清洗和處理股票數據
def clean_and_process_stock_data(tickers, data):
    data_cleaned = {}
    for ticker in tickers:
        df = data[ticker].dropna()
        numeric_cols = df.select_dtypes(include='number')
        
        mean = numeric_cols.mean()
        std = numeric_cols.std()
        threshold = 3  
        
        outliers = (numeric_cols - mean).abs() > threshold * std
        df_cleaned = numeric_cols.mask(outliers, other=mean, axis=1)
        
        df_cleaned.index = pd.to_datetime(df_cleaned.index)
        df_cleaned = df_cleaned.astype(float)
        data_cleaned[ticker] = df_cleaned
    return data_cleaned

# 顯示股票走勢圖
def plot_stock_trend(df, ticker):
    st.line_chart(df['Close'])

# 主函數，用於在 Streamlit 中顯示結果
def main():
    st.title("匯率與股票數據實時更新")

    # 顯示側邊欄選項讓用戶選擇查詢的數據
    choice = st.sidebar.selectbox("選擇要查詢的數據", ("匯率", "股票"))
    
    if choice == "匯率":
        
        # 如果沒有輸入貨幣對，顯示銀行匯率的表格
        currencies_input = st.sidebar.text_input('輸入要查詢的貨幣對 (例:"USDTWD=X", "EURTWD=X")').strip()
        
        if not currencies_input:
            # 查詢銀行匯率並顯示表格
            exchange_rate_data = get_exchange_rates()
                        
            if not exchange_rate_data.empty:
                st.dataframe(exchange_rate_data)  # 顯示所有銀行匯率資料
            
            # 自動刷新頁面每 5 分鐘
            time.sleep(300)  # 5 分鐘後重新加載
            st.rerun()  # 重新加載頁面
        
        else:
            # 查詢單一貨幣對的匯率並顯示折線圖
            currencies = currencies_input.split(",")  # 輸入貨幣對
            currencies = [currency.strip().upper() for currency in currencies if currency.strip()]
            
            exchange_rate_data = get_exchange_rate_data(currencies)

            if not exchange_rate_data.empty:
                st.dataframe(exchange_rate_data)  # 顯示所有匯率資料
                plot_exchange_rate_trend(exchange_rate_data, currencies)  # 顯示折線圖

    elif choice == "股票":
        # 顯示股票代碼輸入框
        tickers_input = st.text_input("請輸入股票代碼 ( 多個代碼用','分隔，例: 0050.TW, 0057.TW, 0056.TW ) ").strip()

        if tickers_input:
            tickers = [ticker.strip() for ticker in tickers_input.split(',') if ticker.strip()]
            
            # 呼叫 get_valid_tickers 來驗證股票代碼是否有效
            valid_tickers = get_valid_tickers(tickers)

            if valid_tickers:
                # 顯示股票走勢圖
                st.write("顯示股票走勢圖")
                
                # 請求股票數據
                start_date, end_date = get_date_range()  # 獲取日期範圍
                data = yf.download(valid_tickers, start=start_date, end=end_date, group_by='ticker')

                # 顯示每個股票的走勢圖
                for ticker in valid_tickers:
                    st.write(f"{ticker} 股票走勢圖")
                    plot_stock_trend(data[ticker], ticker)
            else:
                st.warning("請輸入有效的股票代碼")
        else:
            # 如果沒有輸入股票代碼，不進行任何操作
            st.write("請輸入股票代碼以查詢走勢圖")
            
if __name__ == "__main__":
    main()
