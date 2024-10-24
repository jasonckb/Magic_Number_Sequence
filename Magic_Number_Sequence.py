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

def safe_compare(a, b, operator='<='):
    """Safely compare two values that might be Series"""
    try:
        a_val = float(a)
        b_val = float(b)
        if operator == '<=':
            return bool(a_val <= b_val)
        elif operator == '>=':
            return bool(a_val >= b_val)
        elif operator == '<':
            return bool(a_val < b_val)
        elif operator == '>':
            return bool(a_val > b_val)
        elif operator == '==':
            return bool(a_val == b_val)
    except:
        return False

def safe_minmax(values, operation='min'):
    """Safely compute min or max of values that might be Series"""
    try:
        float_values = [float(v) for v in values]
        if operation == 'min':
            return min(float_values)
        else:  # max
            return max(float_values)
    except:
        return float('-inf') if operation == 'min' else float('inf')

# Update the check functions:
def check_buy_flip(df, i):
    if i < 5:
        return False
    return safe_compare(df['Close'].iloc[i], df['Close'].iloc[i-4], '<') and \
           safe_compare(df['Close'].iloc[i-1], df['Close'].iloc[i-5], '>')

def check_sell_flip(df, i):
    if i < 5:
        return False
    return safe_compare(df['Close'].iloc[i], df['Close'].iloc[i-4], '>') and \
           safe_compare(df['Close'].iloc[i-1], df['Close'].iloc[i-5], '<')

def check_buy_setup(df, i):
    if i < 4:
        return False
    return safe_compare(df['Close'].iloc[i], df['Close'].iloc[i-4], '<')

def check_sell_setup(df, i):
    if i < 4:
        return False
    return safe_compare(df['Close'].iloc[i], df['Close'].iloc[i-4], '>')

def check_buy_countdown(df, i):
    if i < 2:
        return False
    return safe_compare(df['Close'].iloc[i], df['Low'].iloc[i-2], '<=')

def check_sell_countdown(df, i):
    if i < 2:
        return False
    return safe_compare(df['Close'].iloc[i], df['High'].iloc[i-2], '>=')

def check_buy_perfection(df, setup_start, i):
    if i - setup_start < 8:
        return False
    bar8_9_low = safe_minmax([df['Low'].iloc[i], df['Low'].iloc[i-1]], 'min')
    bar6_7_low = safe_minmax([df['Low'].iloc[i-3], df['Low'].iloc[i-2]], 'min')
    return safe_compare(bar8_9_low, bar6_7_low, '<=')

def check_sell_perfection(df, setup_start, i):
    if i - setup_start < 8:
        return False
    bar8_9_high = safe_minmax([df['High'].iloc[i], df['High'].iloc[i-1]], 'max')
    bar6_7_high = safe_minmax([df['High'].iloc[i-3], df['High'].iloc[i-2]], 'max')
    return safe_compare(bar8_9_high, bar6_7_high, '>=')

def get_tdst_level(df, setup_start_idx, end_idx, is_buy_setup):
    if is_buy_setup:
        prior_close = float(df['Close'].iloc[setup_start_idx-1]) if setup_start_idx > 0 else float('-inf')
        highest_high = safe_minmax(df['High'].iloc[setup_start_idx:end_idx+1], 'max')
        return safe_minmax([prior_close, highest_high], 'max')
    else:
        prior_close = float(df['Close'].iloc[setup_start_idx-1]) if setup_start_idx > 0 else float('inf')
        lowest_low = safe_minmax(df['Low'].iloc[setup_start_idx:end_idx+1], 'min')
        return safe_minmax([prior_close, lowest_low], 'min')
        
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
        # Check TDST violations
        if buy_countdown_active and tdst.check_resistance_violation(df['Close'].iloc[i]):
            buy_countdown_active = False
            buy_setup_count = 0
            buy_countdown_bars = []
            
        if sell_countdown_active and tdst.check_support_violation(df['Close'].iloc[i]):
            sell_countdown_active = False
            sell_setup_count = 0
            sell_countdown_bars = []
        
        # Setup flips
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
        
        # Buy setup phase
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
        
        # Sell setup phase
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
        
        # Buy countdown phase
        if buy_setup_complete and not buy_countdown_active and not need_new_buy_setup:
            if safe_compare(df['Close'].iloc[i], df['Low'].iloc[i-2], '<='):
                buy_countdown_active = True
                buy_setup_complete = False
                buy_countdown[i] = 1
                buy_setup_count = 1
                buy_countdown_bars = [i]
                waiting_for_buy_13 = False
                
        elif buy_countdown_active:
            if waiting_for_buy_13:
                if safe_compare(df['Low'].iloc[i], bar8_close_buy, '<='):
                    buy_countdown[i] = 13
                    buy_countdown_active = False
                    waiting_for_buy_13 = False
                    buy_countdown_bars = []
                    need_new_buy_setup = True
                elif safe_compare(df['Close'].iloc[i], df['Low'].iloc[i-2], '<='):
                    buy_countdown[i] = 12
                    buy_deferred[i] = True
                else:
                    buy_countdown[i] = 12
            else:
                if safe_compare(df['Close'].iloc[i], df['Low'].iloc[i-2], '<='):
                    buy_countdown_bars.append(i)
                    
                    if buy_setup_count < 12:
                        buy_setup_count += 1
                        buy_countdown[i] = buy_setup_count
                        if buy_setup_count == 12:
                            if len(buy_countdown_bars) >= 8:
                                bar8_idx = buy_countdown_bars[-8]
                                bar8_close_buy = float(df['Close'].iloc[bar8_idx])
                                waiting_for_buy_13 = True
        
        # Sell countdown phase
        if sell_setup_complete and not sell_countdown_active and not need_new_sell_setup:
            if safe_compare(df['Close'].iloc[i], df['High'].iloc[i-2], '>='):
                sell_countdown_active = True
                sell_setup_complete = False
                sell_countdown[i] = 1
                sell_setup_count = 1
                sell_countdown_bars = [i]
                waiting_for_sell_13 = False
                
        elif sell_countdown_active:
            if waiting_for_sell_13:
                if safe_compare(df['High'].iloc[i], bar8_close_sell, '>='):
                    sell_countdown[i] = 13
                    sell_countdown_active = False
                    waiting_for_sell_13 = False
                    sell_countdown_bars = []
                    need_new_sell_setup = True
                elif safe_compare(df['Close'].iloc[i], df['High'].iloc[i-2], '>='):
                    sell_countdown[i] = 12
                    sell_deferred[i] = True
                else:
                    sell_countdown[i] = 12
            else:
                if safe_compare(df['Close'].iloc[i], df['High'].iloc[i-2], '>='):
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
    if df.empty:
        st.error("No data available for the selected date range")
        return None
        
    buy_setup, sell_setup, buy_countdown, sell_countdown, buy_perfection, sell_perfection, buy_deferred, sell_deferred, tdst = calculate_td_sequential(df)
    buy_setup, sell_setup = clean_incomplete_setups(buy_setup, sell_setup)
    
    # Filter data
    mask = (df.index >= start_date) & (df.index <= end_date)
    df_filtered = df[mask].copy()
    
    if df_filtered.empty:
        st.error("No data available for the selected date range")
        return None

    # Create figure
    fig = go.Figure()

    # Add SINGLE candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=list(df_filtered.index),  # Convert to list
            open=df_filtered['Open'],
            high=df_filtered['High'],
            low=df_filtered['Low'],
            close=df_filtered['Close'],
            name=ticker,
            increasing_line_color='#26A69A',
            decreasing_line_color='#EF5350'
        )
    )

    # Calculate price range for y-axis
    price_range = df_filtered['High'].max() - df_filtered['Low'].min()
    y_min = float(df_filtered['Low'].min() - (price_range * 0.1))
    y_max = float(df_filtered['High'].max() + (price_range * 0.1))

    # Add TDST levels as horizontal lines
    for i, (level, date) in enumerate(tdst.resistance_levels):
        if date in df_filtered.index:
            date_idx = df_filtered.index.get_loc(date)
            end_idx = min(date_idx + 30, len(df_filtered.index) - 1)
            end_date = df_filtered.index[end_idx]
            
            fig.add_shape(
                type="line",
                x0=date,
                x1=end_date,
                y0=level,
                y1=level,  # Same y value for horizontal line
                line=dict(color="red", width=1, dash="dash"),
                xref='x',
                yref='y'
            )
            
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
        if date in df_filtered.index:
            date_idx = df_filtered.index.get_loc(date)
            end_idx = min(date_idx + 30, len(df_filtered.index) - 1)
            end_date = df_filtered.index[end_idx]
            
            fig.add_shape(
                type="line",
                x0=date,
                x1=end_date,
                y0=level,
                y1=level,  # Same y value for horizontal line
                line=dict(color="green", width=1, dash="dash"),
                xref='x',
                yref='y'
            )
            
            fig.add_annotation(
                x=end_date,
                y=level,
                text=f"TrendLine S{len(tdst.support_levels)-i}",
                showarrow=False,
                xanchor="left",
                xshift=10,
                font=dict(color="green", size=10)
            )

    # Add annotations for setups and countdowns [annotation code remains the same]

    # Update layout ONCE
    fig.update_layout(
        title=f'Magic Number Sequence Analysis - {ticker}',
        yaxis=dict(
            title='Price',
            side='right',
            range=[y_min, y_max],
            autorange=False,
            showgrid=True,
            gridcolor='lightgrey'
        ),
        xaxis=dict(
            title='Date',
            rangeslider=dict(visible=False),
            showgrid=True,
            gridcolor='lightgrey',
            type='category'  # Important for proper display
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=800,
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig

def main():
    try:
        st.write("Starting data download...")
        df, display_start, end_date = download_data()
        
        if df.empty:
            st.error("No data available for the selected stock")
            return
            
        st.write("Data loaded successfully")
        st.write(f"Data shape: {df.shape}")
        
        # Create chart
        fig = create_td_sequential_chart(df, display_start, end_date)
        if fig is not None:
            st.write("Chart created successfully")
            st.plotly_chart(
                fig, 
                use_container_width=True, 
                config={
                    'scrollZoom': True,
                    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape']
                }
            )
        else:
            st.error("Failed to create chart - returned None")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.write("Full error:", traceback.format_exc())
