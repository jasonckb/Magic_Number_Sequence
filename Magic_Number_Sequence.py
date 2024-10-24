import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from collections import deque

# Set page config
st.set_page_config(
    page_title="Magic Number Sequence by Jason Chan",
    layout="wide"
)

# Title
st.title("Magic Number Sequence by Jason Chan")

# Sidebar for ticker input
with st.sidebar:
    ticker = st.text_input("Enter Stock Symbol", "AAPL")
    
    # Process HK stocks
    if ticker.isdigit():
        ticker = f"{int(ticker):04d}.HK"

def download_data():
    # Calculate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=500)  # Download 500 days
    display_start = end_date - timedelta(days=365)  # Show 1 year
    
    # Convert to date objects
    display_start = display_start.date()
    end_date = end_date.date()
    start_date = start_date.date()
    
    # Download data
    df = yf.download(ticker, start=start_date, end=end_date)
    df.index = df.index.date
    
    return df, display_start, end_date

def check_buy_flip(df, i):
    if i < 5:
        return False
    current_condition = df['Close'].iloc[i] < df['Close'].iloc[i-4]
    prev_condition = df['Close'].iloc[i-1] > df['Close'].iloc[i-5]
    return bool(current_condition and prev_condition)  # Convert to boolean

def check_sell_flip(df, i):
    if i < 5:
        return False
    current_condition = df['Close'].iloc[i] > df['Close'].iloc[i-4]
    prev_condition = df['Close'].iloc[i-1] < df['Close'].iloc[i-5]
    return bool(current_condition and prev_condition)  # Convert to boolean

def check_buy_setup(df, i):
    if i < 4:
        return False
    return bool(df['Close'].iloc[i] < df['Close'].iloc[i-4])  # Convert to boolean

def check_sell_setup(df, i):
    if i < 4:
        return False
    return bool(df['Close'].iloc[i] > df['Close'].iloc[i-4])  # Convert to boolean

def check_buy_countdown(df, i):
    if i < 2:
        return False
    return bool(df['Close'].iloc[i] <= df['Low'].iloc[i-2])  # Convert to boolean

def check_sell_countdown(df, i):
    if i < 2:
        return False
    return bool(df['Close'].iloc[i] >= df['High'].iloc[i-2])  # Convert to boolea

def check_buy_perfection(df, setup_start, i):
    if i - setup_start < 8:
        return False
    # Convert Series values to floats before comparison
    bar8_9_low = min(float(df['Low'].iloc[i]), float(df['Low'].iloc[i-1]))
    bar6_7_low = min(float(df['Low'].iloc[i-3]), float(df['Low'].iloc[i-2]))
    return bool(bar8_9_low <= bar6_7_low)

def check_sell_perfection(df, setup_start, i):
    if i - setup_start < 8:
        return False
    # Convert Series values to floats before comparison
    bar8_9_high = max(float(df['High'].iloc[i]), float(df['High'].iloc[i-1]))
    bar6_7_high = max(float(df['High'].iloc[i-3]), float(df['High'].iloc[i-2]))
    return bool(bar8_9_high >= bar6_7_high)

def get_tdst_level(df, setup_start_idx, end_idx, is_buy_setup):
    if is_buy_setup:
        prior_close = float(df['Close'].iloc[setup_start_idx-1]) if setup_start_idx > 0 else float('-inf')
        highest_high = float(df['High'].iloc[setup_start_idx:end_idx+1].max())
        return float(max(prior_close, highest_high))
    else:
        prior_close = float(df['Close'].iloc[setup_start_idx-1]) if setup_start_idx > 0 else float('inf')
        lowest_low = float(df['Low'].iloc[setup_start_idx:end_idx+1].min())
        return float(min(prior_close, lowest_low))


def clean_incomplete_setups(buy_setup, sell_setup):
    cleaned_buy = np.zeros_like(buy_setup)
    cleaned_sell = np.zeros_like(sell_setup)
    
    # Process buy setups
    i = 0
    while i < len(buy_setup):
        if buy_setup[i] > 0:
            start = i
            while i < len(buy_setup) and buy_setup[i] > 0:
                i += 1
            end = i
            
            if buy_setup[end-1] == 9:
                cleaned_buy[start:end] = buy_setup[start:end]
        else:
            i += 1
    
    # Process sell setups
    i = 0
    while i < len(sell_setup):
        if sell_setup[i] > 0:
            start = i
            while i < len(sell_setup) and sell_setup[i] > 0:
                i += 1
            end = i
            
            if sell_setup[end-1] == 9:
                cleaned_sell[start:end] = sell_setup[start:end]
        else:
            i += 1
    
    return cleaned_buy, cleaned_sell

class TDSTLevels:
    def __init__(self):
        self.resistance_levels = []
        self.support_levels = []
        self.active_resistance = None
        self.active_support = None
    
    def add_resistance(self, price, date):
        self.resistance_levels.append((float(price), date))  # Convert to float
        self.active_resistance = float(price)
    
    def add_support(self, price, date):
        self.support_levels.append((float(price), date))  # Convert to float
        self.active_support = float(price)
    
    def check_resistance_violation(self, price):
        if self.active_resistance is not None:
            return float(price) > self.active_resistance  # Convert to float
        return False
    
    def check_support_violation(self, price):
        if self.active_support is not None:
            return float(price) < self.active_support  # Convert to float
        return False

def calculate_td_sequential(df):
    buy_setup = np.zeros(len(df))
    sell_setup = np.zeros(len(df))
    buy_countdown = np.zeros(len(df))
    sell_countdown = np.zeros(len(df))
    buy_perfection = np.zeros(len(df))
    sell_perfection = np.zeros(len(df))
    buy_deferred = np.zeros(len(df), dtype=bool)
    sell_deferred = np.zeros(len(df), dtype=bool)
    buy_setup_active = False
    sell_setup_active = False
    buy_countdown_active = False
    sell_countdown_active = False
    buy_setup_complete = False
    sell_setup_complete = False
    setup_start_idx = 0
    
    buy_countdown_bars = []
    sell_countdown_bars = []
    
    tdst = TDSTLevels()
    
    need_new_buy_setup = False
    need_new_sell_setup = False
    
    buy_setup_count = 0
    sell_setup_count = 0
    
    waiting_for_buy_13 = False
    waiting_for_sell_13 = False
    bar8_close_buy = None
    bar8_close_sell = None
    
    for i in range(len(df)):
        if buy_countdown_active and tdst.check_resistance_violation(float(df['Close'].iloc[i])):
            buy_countdown_active = False
            buy_setup_count = 0
            buy_countdown_bars = []
            
        if sell_countdown_active and tdst.check_support_violation(float(df['Close'].iloc[i])):
            sell_countdown_active = False
            sell_setup_count = 0
            sell_countdown_bars = []
        
        if check_buy_flip(df, i):
            buy_setup_active = True
            sell_setup_active = False
            setup_start_idx = i
            buy_setup[i] = 1
        elif check_sell_flip(df, i):
            sell_setup_active = True
            buy_setup_active = False
            setup_start_idx = i
            sell_setup[i] = 1
        
        if buy_setup_active:
            if check_buy_setup(df, i):
                if i > 0 and buy_setup[i-1] > 0:
                    current_count = buy_setup[i-1] + 1
                    if current_count <= 9:
                        buy_setup[i] = current_count
                        if current_count == 9:
                            if check_buy_perfection(df, setup_start_idx, i):
                                buy_perfection[i] = 1
                            buy_setup_active = False
                            buy_setup_complete = True
                            need_new_buy_setup = False
                            resistance = get_tdst_level(df, setup_start_idx, i, True)
                            tdst.add_resistance(resistance, df.index[i])
                else:
                    buy_setup[i] = 1
            else:
                buy_setup_active = False
        
        if sell_setup_active:
            if check_sell_setup(df, i):
                if i > 0 and sell_setup[i-1] > 0:
                    current_count = sell_setup[i-1] + 1
                    if current_count <= 9:
                        sell_setup[i] = current_count
                        if current_count == 9:
                            if check_sell_perfection(df, setup_start_idx, i):
                                sell_perfection[i] = 1
                            sell_setup_active = False
                            sell_setup_complete = True
                            need_new_sell_setup = False
                            support = get_tdst_level(df, setup_start_idx, i, False)
                            tdst.add_support(support, df.index[i])
                else:
                    sell_setup[i] = 1
            else:
                sell_setup_active = False
        
        if buy_setup_complete and not buy_countdown_active and not need_new_buy_setup:
            if bool(df['Close'].iloc[i] <= df['Low'].iloc[i-2]):  # Convert to boolean
                buy_countdown_active = True
                buy_setup_complete = False
                buy_countdown[i] = 1
                buy_setup_count = 1
                buy_countdown_bars = [i]
                waiting_for_buy_13 = False
                
        elif buy_countdown_active:
            if waiting_for_buy_13:
                if bool(df['Low'].iloc[i] <= bar8_close_buy):  # Convert to boolean
                    buy_countdown[i] = 13
                    buy_countdown_active = False
                    waiting_for_buy_13 = False
                    buy_countdown_bars = []
                    need_new_buy_setup = True
                elif bool(df['Close'].iloc[i] <= df['Low'].iloc[i-2]):  # Convert to boolean
                    buy_countdown[i] = 12
                    buy_deferred[i] = True
                else:
                    buy_countdown[i] = 12
            else:
                if bool(df['Close'].iloc[i] <= df['Low'].iloc[i-2]):  # Convert to boolean
                    buy_countdown_bars.append(i)
                    
                    if buy_setup_count < 12:
                        buy_setup_count += 1
                        buy_countdown[i] = buy_setup_count
                        if buy_setup_count == 12:
                            if len(buy_countdown_bars) >= 8:
                                bar8_idx = buy_countdown_bars[-8]
                                bar8_close_buy = float(df['Close'].iloc[bar8_idx])  # Convert to float
                                waiting_for_buy_13 = True
        
        if sell_setup_complete and not sell_countdown_active and not need_new_sell_setup:
            if bool(df['Close'].iloc[i] >= df['High'].iloc[i-2]):
                sell_countdown_active = True
                sell_setup_complete = False
                sell_countdown[i] = 1
                sell_setup_count = 1
                sell_countdown_bars = [i]
                waiting_for_sell_13 = False
                
        elif sell_countdown_active:
            if waiting_for_sell_13:
                if bool(df['High'].iloc[i] >= float(bar8_close_sell)):
                    sell_countdown[i] = 13
                    sell_countdown_active = False
                    waiting_for_sell_13 = False
                    sell_countdown_bars = []
                    need_new_sell_setup = True
                elif bool(df['Close'].iloc[i] >= df['High'].iloc[i-2]):
                    sell_countdown[i] = 12
                    sell_deferred[i] = True
                else:
                    sell_countdown[i] = 12
            else:
                if bool(df['Close'].iloc[i] >= df['High'].iloc[i-2]):
                    sell_countdown_bars.append(i)
                    
                    if sell_setup_count < 12:
                        sell_setup_count += 1
                        sell_countdown[i] = sell_setup_count
                        if sell_setup_count == 12:
                            if len(sell_countdown_bars) >= 8:
                                bar8_idx = sell_countdown_bars[-8]
                                bar8_close_sell = float(df['Close'].iloc[bar8_idx])
                                waiting_for_sell_13 = True
    
    return buy_setup, sell_setup, buy_countdown, sell_countdown, buy_perfection, sell_perfection, buy_deferred, sell_deferred, tdst

def create_td_sequential_chart(df, start_date, end_date):
    buy_setup, sell_setup, buy_countdown, sell_countdown, buy_perfection, sell_perfection, buy_deferred, sell_deferred, tdst = calculate_td_sequential(df)
    
    buy_setup, sell_setup = clean_incomplete_setups(buy_setup, sell_setup)
    
    mask = (df.index >= start_date) & (df.index <= end_date)
    df_filtered = df[mask]
    
    fig = go.Figure(data=[go.Candlestick(x=df_filtered.index,
                                        open=df_filtered['Open'],
                                        high=df_filtered['High'],
                                        low=df_filtered['Low'],
                                        close=df_filtered['Close'])])
    
    # Add TrendLine levels (formerly TDST)
    for i, (level, date) in enumerate(tdst.resistance_levels):
        date_idx = df_filtered.index.get_loc(date) if date in df_filtered.index else -1
        if date_idx != -1:
            end_idx = min(date_idx + 30, len(df_filtered) - 1)
            end_date = df_filtered.index[end_idx]
            
            fig.add_shape(
                type="line",
                x0=date,
                x1=end_date,
                y0=level,
                y1=level,
                line=dict(
                    color="red",
                    width=1,
                    dash="dash"
                )
            )
    
    for i, (level, date) in enumerate(tdst.support_levels):
        date_idx = df_filtered.index.get_loc(date) if date in df_filtered.index else -1
        if date_idx != -1:
            end_idx = min(date_idx + 30, len(df_filtered) - 1)
            end_date = df_filtered.index[end_idx]
            
            fig.add_shape(
                type="line",
                x0=date,
                x1=end_date,
                y0=level,
                y1=level,
                line=dict(
                    color="green",
                    width=1,
                    dash="dash"
                )
            )
    
    # Add annotations
    for i in range(len(df_filtered)):
        idx = df_filtered.index[i]
        orig_idx = df.index.get_loc(idx)
        
        # Buy setup counts
        if buy_setup[orig_idx] > 0:
            count_text = str(int(buy_setup[orig_idx]))
            font_size = 13 if buy_setup[orig_idx] == 9 else 10
            if buy_setup[orig_idx] == 9:
                if buy_perfection[orig_idx]:
                    count_text += "↑"
                else:
                    count_text += "+"
            fig.add_annotation(x=idx, y=df_filtered['Low'].iloc[i],
                             text=count_text,
                             showarrow=False,
                             yshift=-10,
                             font=dict(color="green", size=font_size))
        
        # Buy countdown counts
        if buy_countdown[orig_idx] > 0 or buy_deferred[orig_idx]:
            if buy_deferred[orig_idx]:
                count_text = "+"
            else:
                count_text = str(int(buy_countdown[orig_idx]))
            font_size = 13 if buy_countdown[orig_idx] == 13 else 10
            fig.add_annotation(x=idx, y=df_filtered['Low'].iloc[i],
                             text=count_text,
                             showarrow=False,
                             yshift=-25,
                             font=dict(color="red", size=font_size))
        
        # Sell setup counts
        if sell_setup[orig_idx] > 0:
            count_text = str(int(sell_setup[orig_idx]))
            font_size = 13 if sell_setup[orig_idx] == 9 else 10
            if sell_setup[orig_idx] == 9:
                if sell_perfection[orig_idx]:
                    count_text += "↓"
                else:
                    count_text += "+"
            fig.add_annotation(x=idx, y=df_filtered['High'].iloc[i],
                             text=count_text,
                             showarrow=False,
                             yshift=10,
                             font=dict(color="green", size=font_size))
        
        # Sell countdown counts
        if sell_countdown[orig_idx] > 0 or sell_deferred[orig_idx]:
            if sell_deferred[orig_idx]:
                count_text = "+"
            else:
                count_text = str(int(sell_countdown[orig_idx]))
            font_size = 13 if sell_countdown[orig_idx] == 13 else 10
            fig.add_annotation(x=idx, y=df_filtered['High'].iloc[i],
                             text=count_text,
                             showarrow=False,
                             yshift=25,
                             font=dict(color="red", size=font_size))
    
    # Add TrendLine level labels (formerly TDST)
    for i, (level, date) in enumerate(tdst.resistance_levels):
        date_idx = df_filtered.index.get_loc(date) if date in df_filtered.index else -1
        if date_idx != -1:
            end_idx = min(date_idx + 30, len(df_filtered) - 1)
            end_date = df_filtered.index[end_idx]
            
            fig.add_annotation(
                x=end_date,
                y=level,
                text=f"TrendLine R{len(tdst.resistance_levels)-i}",
                showarrow=False,
                xanchor="left",
                xshift=10,
                font=dict(color="red", size=10)
            )
    
    for i, (level, date) in enumerate(tdst.support_levels):
        date_idx = df_filtered.index.get_loc(date) if date in df_filtered.index else -1
        if date_idx != -1:
            end_idx = min(date_idx + 30, len(df_filtered) - 1)
            end_date = df_filtered.index[end_idx]
            
            fig.add_annotation(
                x=end_date,
                y=level,
                text=f"TrendLine S{len(tdst.support_levels)-i}",
                showarrow=False,
                xanchor="left",
                xshift=10,
                font=dict(color="green", size=10)
            )
    
    # Update layout
    fig.update_layout(
        title=f'Magic Number Sequence Analysis - {ticker}',
        yaxis_title='Price',
        xaxis_title='Date',
        showlegend=False,
        height=800,
        width=None,  # Let Streamlit handle the width
        yaxis=dict(
            autorange=True,
            fixedrange=False,
            range=[df_filtered['Low'].min() * 0.99, df_filtered['High'].max() * 1.01]
        ),
        xaxis=dict(
            rangeslider=dict(visible=False),
            type='category',
            tickangle=45,
            tickformat="%Y-%m-%d"
        )
    )
    
    return fig

def main():
    try:
        df, display_start, end_date = download_data()
        fig = create_td_sequential_chart(df, display_start, end_date)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

if __name__ == "__main__":
    main()
