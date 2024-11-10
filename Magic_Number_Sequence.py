import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import numpy as np
import requests
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(layout="wide", page_title="Magic Number Sequence")
st.title("Magic Number Sequence by Jason Chan")


def get_stocks_from_github(asset_type):
    """Get stock list for the specified asset type"""
    # Default stock lists
    default_stocks = {
         "HK Stocks": ["^HSI",
            "0001.HK", "0003.HK", "0005.HK", "0006.HK", "0011.HK", "0012.HK", "0016.HK", "0017.HK",
            "0019.HK", "0020.HK", "0027.HK", "0066.HK", "0175.HK", "0241.HK", "0267.HK", "0268.HK",
            "0285.HK", "0288.HK", "0291.HK", "0293.HK", "0358.HK", "0386.HK", "0388.HK", "0522.HK",
            "0669.HK", "0688.HK", "0700.HK", "0762.HK", "0772.HK", "0799.HK", "0823.HK", "0836.HK",
            "0853.HK", "0857.HK", "0868.HK", "0883.HK", "0909.HK", "0914.HK", "0916.HK", "0939.HK",
            "0941.HK", "0960.HK", "0968.HK", "0981.HK", "0992.HK", "1024.HK", "1038.HK", "1044.HK",
            "1093.HK", "1109.HK", "1113.HK", "1177.HK", "1211.HK", "1299.HK", "1347.HK", "1398.HK",
            "1772.HK", "1776.HK", "1787.HK", "1801.HK", "1810.HK", "1818.HK", "1833.HK", "1876.HK",
            "1898.HK", "1928.HK", "1929.HK", "1997.HK", "2007.HK", "2013.HK", "2015.HK", "2018.HK",
            "2269.HK", "2313.HK", "2318.HK", "2319.HK", "2331.HK", "2333.HK", "2382.HK", "2388.HK",
            "2518.HK", "2628.HK", "3690.HK", "3888.HK", "3888.HK", "3968.HK", "6060.HK", "6078.HK",
            "6098.HK", "6618.HK", "6690.HK", "6862.HK", "9618.HK", "9626.HK", "9698.HK", "9888.HK",
            "9961.HK", "9988.HK", "9999.HK"
         ],
        "US Stocks": ["^NDX", "^SPX",
            "AAPN", "ABBV", "ABNB", "ACN", "ADBE", "AMD", "AMGN", "AMZN", "AMT", "ASML",
            "AVGO", "BA", "BKNG", "BLK", "CAT", "CCL", "CDNS", "CEG", "CHTR", "COST", 
            "CRM", "CRWD", "CVS", "CVX", "DDOG", "DE", "DIS", "EQIX", "FTNT", "GE",
            "GILD", "GOOG", "GS", "HD", "IBM", "ICE", "IDXX", "INTC", "INTU", "ISRG",
            "JNJ", "JPM", "KO", "LEN", "LLY", "LRCX", "MA", "META", "MMM", "MRK", 
            "MS", "MSFT", "MU", "NEE", "NFLX", "NRG", "NVO", "NVDA", "OXY", "PANW",
            "PFE", "PG", "PGR", "PLTR", "PYPL", "QCOM", "REGN", "SBUX", "SMH", "SNOW",
            "SPGI", "TEAM", "TJX", "TRAV", "TSM", "TSLA", "TTD", "TXN", "UNH", "UPS",
            "V", "VST", "VZ", "WMT", "XOM", "ZS",
            "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLRE", "XLY"
        ],
        "World Index": [
            "^SPX", "^NDX", "^RUT", "^SOX", "^TNX", "^DJI", "^HSI", "3032.HK", "XIN9.FGI", 
            "^N225", "^BSESN", "^KS11", "^TWII", "^GDAXI", "^FTSE", "^FCHI", "^BVSP", "EEMA", 
            "EEM", "^HUI", "CL=F", "GC=F", "HG=F", "SI=F", "DX-Y.NYB", "BTC=F", "ETH=F"
        ]
    }
    
    try:
        return default_stocks[asset_type]
    except KeyError:
        st.error(f"Unknown asset type: {asset_type}")
        return []


def check_ticker_format(ticker):
    """Format ticker symbol for Hong Kong stocks"""
    if ticker.isdigit():
        return ticker.zfill(4) + ".HK"
    return ticker.upper()

def clean_yahoo_data(df):
    """Clean and format Yahoo Finance data"""
    try:
        if isinstance(df.columns, pd.MultiIndex):
            base_cols = df.columns.get_level_values(0).unique()
            ticker = df.columns.get_level_values(1)[0]
            df_cleaned = pd.DataFrame(index=df.index)
            for col in base_cols:
                df_cleaned[col] = df[(col, ticker)]
        else:
            df_cleaned = df.copy()
        df_cleaned = df_cleaned.dropna(subset=['Open', 'High', 'Low', 'Close'])
        return df_cleaned
    except Exception as e:
        st.error(f"Error in data cleaning: {str(e)}")
        return pd.DataFrame()

def safe_compare(a, b, operator='<='):
    """Safe comparison of values"""
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
    """Safe min/max calculation"""
    try:
        float_values = [float(v) for v in values]
        if operation == 'min':
            return min(float_values)
        else:
            return max(float_values)
    except:
        return float('-inf') if operation == 'min' else float('inf')

# TD Sequential Check Functions
def check_buy_flip(df, i):
    """Check for valid buy price flip"""
    if i < 5:
        return False
    return safe_compare(df['Close'].iloc[i], df['Close'].iloc[i-4], '<') and \
           safe_compare(df['Close'].iloc[i-1], df['Close'].iloc[i-5], '>')

def check_sell_flip(df, i):
    """Check for valid sell price flip"""
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

def check_bar8_rule(df, current_idx, bar8_idx, is_buy_countdown):
    if is_buy_countdown:
        return safe_compare(df['Low'].iloc[current_idx], df['Close'].iloc[bar8_idx], '<=')
    else:
        return safe_compare(df['High'].iloc[current_idx], df['Close'].iloc[bar8_idx], '>=')

def check_18_bar_rule(df, current_idx, last_flip_idx, is_buy_countdown):
    if last_flip_idx < 0 or current_idx - last_flip_idx < 18:
        return False
    
    for i in range(last_flip_idx, current_idx + 1):
        if is_buy_countdown and check_buy_flip(df, i):
            return False
        elif not is_buy_countdown and check_sell_flip(df, i):
            return False
    return True

class TDSTLevels:
    def __init__(self):
        self.resistance_levels = []
        self.support_levels = []
        self.active_resistance = None
        self.active_support = None
    
    def add_resistance(self, price, date):
        self.resistance_levels.append((float(price), date))
        self.active_resistance = float(price)
    
    def add_support(self, price, date):
        self.support_levels.append((float(price), date))
        self.active_support = float(price)
    
    def check_resistance_violation(self, price):
        if self.active_resistance is not None:
            return safe_compare(price, self.active_resistance, '>')
        return False
    
    def check_support_violation(self, price):
        if self.active_support is not None:
            return safe_compare(price, self.active_support, '<')
        return False

def get_tdst_level(df, setup_start_idx, end_idx, is_buy_setup):
    if is_buy_setup:
        prior_close = float(df['Close'].iloc[setup_start_idx-1]) if setup_start_idx > 0 else float('-inf')
        highest_high = safe_minmax(df['High'].iloc[setup_start_idx:end_idx+1], 'max')
        return safe_minmax([prior_close, highest_high], 'max')
    else:
        prior_close = float(df['Close'].iloc[setup_start_idx-1]) if setup_start_idx > 0 else float('inf')
        lowest_low = safe_minmax(df['Low'].iloc[setup_start_idx:end_idx+1], 'min')
        return safe_minmax([prior_close, lowest_low], 'min')

import numpy as np

def calculate_td_sequential(df):
    """Main TD Sequential calculation function with proper reset of incomplete setups."""
    # Initialize arrays
    buy_setup = np.zeros(len(df))
    sell_setup = np.zeros(len(df))
    buy_countdown = np.zeros(len(df))
    sell_countdown = np.zeros(len(df))
    buy_perfection = np.zeros(len(df))
    sell_perfection = np.zeros(len(df))
    buy_deferred = np.zeros(len(df), dtype=bool)
    sell_deferred = np.zeros(len(df), dtype=bool)
    
    # New arrays to track completed setups
    buy_setup_completed = np.zeros(len(df), dtype=bool)
    sell_setup_completed = np.zeros(len(df), dtype=bool)
    
    # Initialize state variables
    buy_setup_active = False
    sell_setup_active = False
    buy_countdown_active = False
    sell_countdown_active = False
    
    # Initialize setup start indices
    setup_start_idx_buy = None
    setup_start_idx_sell = None
    
    # Initialize other variables
    buy_setup_count = 0
    sell_setup_count = 0
    waiting_for_buy_13 = False
    waiting_for_sell_13 = False
    bar8_close_buy = None
    bar8_close_sell = None
    buy_countdown_bars = []
    sell_countdown_bars = []
    tdst = TDSTLevels()
    last_opposite_flip = {'buy': -1, 'sell': -1}
    
    for i in range(len(df)):
        # Check TDST violations
        if buy_countdown_active and tdst.check_resistance_violation(df['Close'].iloc[i]):
            buy_countdown_active = False
            buy_setup_count = 0
            buy_countdown_bars = []
            waiting_for_buy_13 = False
            bar8_close_buy = None
            
        if sell_countdown_active and tdst.check_support_violation(df['Close'].iloc[i]):
            sell_countdown_active = False
            sell_setup_count = 0
            sell_countdown_bars = []
            waiting_for_sell_13 = False
            bar8_close_sell = None
        
        # Track opposite price flips for 18-bar rule
        if check_buy_flip(df, i) and sell_countdown_active:
            last_opposite_flip['sell'] = i
        if check_sell_flip(df, i) and buy_countdown_active:
            last_opposite_flip['buy'] = i
            
        # Check 18-bar rule
        if buy_countdown_active and check_18_bar_rule(df, i, last_opposite_flip['buy'], True):
            buy_countdown_active = False
            buy_setup_count = 0
            buy_countdown_bars = []
            waiting_for_buy_13 = False
            bar8_close_buy = None
            
        if sell_countdown_active and check_18_bar_rule(df, i, last_opposite_flip['sell'], False):
            sell_countdown_active = False
            sell_setup_count = 0
            sell_countdown_bars = []
            waiting_for_sell_13 = False
            bar8_close_sell = None
        
        # Buy Setup Phase
        if check_buy_flip(df, i):  # Valid buy price flip
            buy_setup_active = True
            setup_start_idx_buy = i
            buy_setup[i] = 1
            # Cancel active sell setup if not completed
            if sell_setup_active and i > 0 and sell_setup[i-1] < 9:
                sell_setup_active = False
                if setup_start_idx_sell is not None:
                    sell_setup[setup_start_idx_sell:i] = 0
                    setup_start_idx_sell = None
        elif buy_setup_active:
            if check_buy_setup(df, i):
                if buy_setup[i-1] > 0:
                    current_count = buy_setup[i-1] + 1
                    buy_setup[i] = current_count
                    if current_count == 9:
                        # Setup completed
                        buy_setup_completed[setup_start_idx_buy:i+1] = True
                        # Check for perfection
                        if check_buy_perfection(df, setup_start_idx_buy, i):
                            buy_perfection[i] = 1
                        # Add TDST resistance level
                        resistance = get_tdst_level(df, setup_start_idx_buy, i, True)
                        tdst.add_resistance(resistance, df.index[i])
                        # Reset variables
                        buy_setup_active = False
                        setup_start_idx_buy = None
                else:
                    buy_setup[i] = 1
            else:
                # Reset incomplete buy setup
                buy_setup_active = False
                if setup_start_idx_buy is not None:
                    buy_setup[setup_start_idx_buy:i] = 0
                    setup_start_idx_buy = None
    
        # Sell Setup Phase
        if check_sell_flip(df, i):  # Valid sell price flip
            sell_setup_active = True
            setup_start_idx_sell = i
            sell_setup[i] = 1
            # Cancel active buy setup if not completed
            if buy_setup_active and i > 0 and buy_setup[i-1] < 9:
                buy_setup_active = False
                if setup_start_idx_buy is not None:
                    buy_setup[setup_start_idx_buy:i] = 0
                    setup_start_idx_buy = None
        elif sell_setup_active:
            if check_sell_setup(df, i):
                if sell_setup[i-1] > 0:
                    current_count = sell_setup[i-1] + 1
                    sell_setup[i] = current_count
                    if current_count == 9:
                        # Setup completed
                        sell_setup_completed[setup_start_idx_sell:i+1] = True
                        # Check for perfection
                        if check_sell_perfection(df, setup_start_idx_sell, i):
                            sell_perfection[i] = 1
                        # Add TDST support level
                        support = get_tdst_level(df, setup_start_idx_sell, i, False)
                        tdst.add_support(support, df.index[i])
                        # Reset variables
                        sell_setup_active = False
                        setup_start_idx_sell = None
                else:
                    sell_setup[i] = 1
            else:
                # Reset incomplete sell setup
                sell_setup_active = False
                if setup_start_idx_sell is not None:
                    sell_setup[setup_start_idx_sell:i] = 0
                    setup_start_idx_sell = None
        
        # Buy Countdown Phase
        if not sell_countdown_active:
            if buy_setup_completed[i] and not buy_countdown_active:
                if i >= 2 and safe_compare(df['Close'].iloc[i], df['Low'].iloc[i-2], '<='):
                    buy_countdown_active = True
                    buy_setup_count = 1
                    buy_countdown[i] = 1
                    buy_countdown_bars = [i]
                    waiting_for_buy_13 = False
                    bar8_close_buy = None
            elif buy_countdown_active:
                if waiting_for_buy_13:
                    if safe_compare(df['Low'].iloc[i], bar8_close_buy, '<='):
                        buy_countdown[i] = 13
                        buy_countdown_active = False
                        waiting_for_buy_13 = False
                        bar8_close_buy = None
                    else:
                        if safe_compare(df['Close'].iloc[i], df['Low'].iloc[i-2], '<='):
                            buy_deferred[i] = True
                else:
                    if safe_compare(df['Close'].iloc[i], df['Low'].iloc[i-2], '<='):
                        buy_setup_count += 1
                        buy_countdown[i] = buy_setup_count
                        buy_countdown_bars.append(i)
                        if buy_setup_count == 8:
                            bar8_close_buy = float(df['Close'].iloc[i])
                        elif buy_setup_count == 13:
                            buy_countdown_active = False
                            waiting_for_buy_13 = False
                            bar8_close_buy = None
                    else:
                        buy_deferred[i] = True
        
        # Sell Countdown Phase
        if not buy_countdown_active:
            if sell_setup_completed[i] and not sell_countdown_active:
                if i >= 2 and safe_compare(df['Close'].iloc[i], df['High'].iloc[i-2], '>='):
                    sell_countdown_active = True
                    sell_setup_count = 1
                    sell_countdown[i] = 1
                    sell_countdown_bars = [i]
                    waiting_for_sell_13 = False
                    bar8_close_sell = None
            elif sell_countdown_active:
                if waiting_for_sell_13:
                    if safe_compare(df['High'].iloc[i], bar8_close_sell, '>='):
                        sell_countdown[i] = 13
                        sell_countdown_active = False
                        waiting_for_sell_13 = False
                        bar8_close_sell = None
                    else:
                        if safe_compare(df['Close'].iloc[i], df['High'].iloc[i-2], '>='):
                            sell_deferred[i] = True
                else:
                    if safe_compare(df['Close'].iloc[i], df['High'].iloc[i-2], '>='):
                        sell_setup_count += 1
                        sell_countdown[i] = sell_setup_count
                        sell_countdown_bars.append(i)
                        if sell_setup_count == 8:
                            bar8_close_sell = float(df['Close'].iloc[i])
                        elif sell_setup_count == 13:
                            sell_countdown_active = False
                            waiting_for_sell_13 = False
                            bar8_close_sell = None
                    else:
                        sell_deferred[i] = True
    
    return (
        buy_setup,
        sell_setup,
        buy_countdown,
        sell_countdown,
        buy_perfection,
        sell_perfection,
        buy_deferred,
        sell_deferred,
        tdst,
        buy_setup_completed,
        sell_setup_completed
    )


import plotly.graph_objs as go

def create_td_sequential_chart(data, formatted_ticker):
    """Create interactive chart with TD Sequential indicators, only plotting completed setups."""
    # Calculate TD Sequential indicators
    buy_setup, sell_setup, buy_countdown, sell_countdown, buy_perfection, \
    sell_perfection, buy_deferred, sell_deferred, tdst, \
    buy_setup_completed, sell_setup_completed = calculate_td_sequential(data)
    
    # Create base figure with candlesticks
    fig = go.Figure(data=[
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='OHLC',
            increasing_line_color='#26A69A',
            decreasing_line_color='#EF5350'
        )
    ])

    # Add annotations for TD Sequential numbers
    for i in range(len(data)):
        # Buy setup counts
        if buy_setup[i] > 0 and buy_setup_completed[i]:
            count_text = str(int(buy_setup[i]))
            font_size = 13 if buy_setup[i] == 9 else 10
            if buy_setup[i] == 9:
                count_text += "↑" if buy_perfection[i] else "+"
            fig.add_annotation(
                x=data.index[i],
                y=float(data['Low'].iloc[i]),
                text=count_text,
                showarrow=False,
                yshift=-10,
                font=dict(color="green", size=font_size)
            )

        # Buy countdown counts
        if buy_countdown[i] > 0 or buy_deferred[i]:
            count_text = "+" if buy_deferred[i] else str(int(buy_countdown[i]))
            font_size = 13 if buy_countdown[i] == 13 else 10
            fig.add_annotation(
                x=data.index[i],
                y=float(data['Low'].iloc[i]),
                text=count_text,
                showarrow=False,
                yshift=-25,
                font=dict(color="red", size=font_size)
            )

        # Sell setup counts
        if sell_setup[i] > 0 and sell_setup_completed[i]:
            count_text = str(int(sell_setup[i]))
            font_size = 13 if sell_setup[i] == 9 else 10
            if sell_setup[i] == 9:
                count_text += "↓" if sell_perfection[i] else "+"
            fig.add_annotation(
                x=data.index[i],
                y=float(data['High'].iloc[i]),
                text=count_text,
                showarrow=False,
                yshift=10,
                font=dict(color="green", size=font_size)
            )

        # Sell countdown counts
        if sell_countdown[i] > 0 or sell_deferred[i]:
            count_text = "+" if sell_deferred[i] else str(int(sell_countdown[i]))
            font_size = 13 if sell_countdown[i] == 13 else 10
            fig.add_annotation(
                x=data.index[i],
                y=float(data['High'].iloc[i]),
                text=count_text,
                showarrow=False,
                yshift=25,
                font=dict(color="red", size=font_size)
            )

    # Add TDST levels
    # Plot resistance levels
    for idx, (level, date) in enumerate(tdst.resistance_levels):
        if date in data.index:
            date_idx = data.index.get_loc(date)
            end_idx = min(date_idx + 30, len(data.index) - 1)
            end_date = data.index[end_idx]
            
            fig.add_shape(
                type="line",
                x0=date,
                x1=end_date,
                y0=float(level),
                y1=float(level),
                line=dict(color="red", width=1, dash="dash"),
                layer="below"
            )
            
            fig.add_annotation(
                x=end_date,
                y=float(level),
                text=f"TDST Resistance {idx+1}",
                showarrow=False,
                xanchor="left",
                xshift=10,
                font=dict(color="red", size=10)
            )

    # Plot support levels
    for idx, (level, date) in enumerate(tdst.support_levels):
        if date in data.index:
            date_idx = data.index.get_loc(date)
            end_idx = min(date_idx + 30, len(data.index) - 1)
            end_date = data.index[end_idx]
            
            fig.add_shape(
                type="line",
                x0=date,
                x1=end_date,
                y0=float(level),
                y1=float(level),
                line=dict(color="green", width=1, dash="dash"),
                layer="below"
            )
            
            fig.add_annotation(
                x=end_date,
                y=float(level),
                text=f"TDST Support {idx+1}",
                showarrow=False,
                xanchor="left",
                xshift=10,
                font=dict(color="green", size=10)
            )

    # Update layout
    y_min = float(data['Low'].min())
    y_max = float(data['High'].max())
    y_padding = (y_max - y_min) * 0.1

    fig.update_layout(
        title=f'Magic Number Sequence Analysis - {formatted_ticker}',
        yaxis=dict(
            title='Price',
            range=[y_min - y_padding, y_max + y_padding],
            side='right',
            showgrid=True,
            gridcolor='LightGrey'
        ),
        xaxis=dict(
            title='Date',
            rangeslider=dict(visible=False),
            showgrid=True,
            gridcolor='LightGrey',
            type='date'
        ),
        height=800,
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig


def get_current_phase(df):
    """Get the current phase numbers for a stock"""
    buy_setup, sell_setup, buy_countdown, sell_countdown, _, _, _, _, _ = calculate_td_sequential(df)
    
    current_phases = {
        'Buy Build Up': '-',
        'Sell Build Up': '-',
        'Buy Run Up': '-',
        'Sell Run Up': '-'
    }
    
    # Only look at the last bar
    if len(buy_setup) > 0:
        last_value = buy_setup[-1]
        if last_value > 0:
            current_phases['Buy Build Up'] = str(int(last_value))
    
    if len(sell_setup) > 0:
        last_value = sell_setup[-1]
        if last_value > 0:
            current_phases['Sell Build Up'] = str(int(last_value))
    
    if len(buy_countdown) > 0:
        last_value = buy_countdown[-1]
        if last_value > 0 and last_value < 14:  # Exclude reset value 14
            current_phases['Buy Run Up'] = str(int(last_value))
    
    if len(sell_countdown) > 0:
        last_value = sell_countdown[-1]
        if last_value > 0 and last_value < 14:  # Exclude reset value 14
            current_phases['Sell Run Up'] = str(int(last_value))
    
    return current_phases

def update_dashboard_data(stock_list):
    """Update data for all stocks in the dashboard"""
    dashboard_data = []
    
    for ticker in stock_list:
        formatted_ticker = check_ticker_format(ticker)
        try:
            end_date = pd.Timestamp.today()
            start_date = end_date - pd.DateOffset(days=30)
            data = yf.download(formatted_ticker, start=start_date, end=end_date, progress=False)
            
            if not data.empty:
                data = clean_yahoo_data(data)
                if len(data) > 0:
                    current_price = f"{data['Close'][-1]:.2f}"
                    daily_change = f"{((data['Close'][-1] / data['Close'][-2] - 1) * 100):.2f}%"
                    
                    phases = get_current_phase(data)
                    
                    dashboard_data.append({
                        'Stock': ticker,
                        'Current Price': current_price,
                        'Daily Change': daily_change,
                        **phases
                    })
        except Exception as e:
            st.error(f"Error processing {ticker}: {str(e)}")
    
    return dashboard_data

def create_summary_section(df):
    """Create summary section for 9s and 13s"""
    # Filter stocks with specific values
    build_up_9_buy = df[df['Buy Build Up'] == '9']['Stock'].tolist()
    build_up_9_sell = df[df['Sell Build Up'] == '9']['Stock'].tolist()
    run_up_13_buy = df[df['Buy Run Up'] == '13']['Stock'].tolist()
    run_up_13_sell = df[df['Sell Run Up'] == '13']['Stock'].tolist()
    
    # Create summary DataFrame
    summary_data = {
        'Build Up (9)': {
            'Buy': ', '.join(build_up_9_buy) if build_up_9_buy else '-',
            'Sell': ', '.join(build_up_9_sell) if build_up_9_sell else '-'
        },
        'Run Up (13)': {
            'Buy': ', '.join(run_up_13_buy) if run_up_13_buy else '-',
            'Sell': ', '.join(run_up_13_sell) if run_up_13_sell else '-'
        }
    }
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df

def main():
    # Initialize session state if not exists
    if 'dashboard_data' not in st.session_state:
        st.session_state.dashboard_data = None
    if 'last_ticker' not in st.session_state:
        st.session_state.last_ticker = None
    if 'asset_type' not in st.session_state:
        st.session_state.asset_type = "HK Stocks"
    if 'ticker_input_box' not in st.session_state:
        st.session_state.ticker_input_box = "AAPL"
    
    # Sidebar Configuration
    st.sidebar.title("Controls")
    
    # Asset type selector
    asset_type = st.sidebar.selectbox(
        "Asset Dashboard Option",
        ["HK Stocks", "US Stocks", "World Index"],
        key="asset_type_selector"
    )
    
    # Get stock list for the selected asset type
    assets = get_stocks_from_github(asset_type)
    
    # Function to update ticker input
    def set_ticker(symbol):
        st.session_state.ticker_input_box = symbol
    
    # Create three columns in the sidebar for the symbols
    if assets:
        st.sidebar.markdown("### Portfolio Symbols")
        cols = st.sidebar.columns(3)
        
        # Calculate number of rows needed
        num_symbols = len(assets)
        num_rows = (num_symbols + 2) // 3  # Round up division
        
        # Create buttons in a grid layout
        for row in range(num_rows):
            for col in range(3):
                idx = row * 3 + col
                if idx < num_symbols:
                    symbol = assets[idx]
                    # Use row and column index for unique key
                    if cols[col].button(
                        symbol,
                        key=f"btn_r{row}_c{col}",
                        on_click=set_ticker,
                        args=(symbol,)
                    ):
                        pass
    
    # Ticker input (after symbol buttons so it can be updated by them)
    ticker_input = st.sidebar.text_input(
        "Enter ticker",
        key="ticker_input_box"
    )
    
    refresh_button = st.sidebar.button("Refresh Dashboard", key="refresh_button")
    
    # Reset dashboard data if asset type changes
    if asset_type != st.session_state.asset_type:
        st.session_state.dashboard_data = None
        st.session_state.asset_type = asset_type
    
    # PART 1: Single Stock Chart Analysis
    st.markdown("### Single Stock Analysis")
    
    if ticker_input:
        try:
            formatted_ticker = check_ticker_format(ticker_input)
            st.write(f"Analyzing: {formatted_ticker}")
            
            end_date = pd.Timestamp.today()
            start_date = end_date - pd.DateOffset(years=1)
            data = yf.download(formatted_ticker, start=start_date, end=end_date, progress=False)
            
            if not data.empty:
                data = clean_yahoo_data(data)
                if len(data) > 0:
                    fig = create_td_sequential_chart(data, formatted_ticker)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error analyzing {ticker_input}: {str(e)}")
    
    # PART 2: Asset Dashboard
    st.markdown(f"### {asset_type} Dashboard")
    
    if assets:
        # Initialize dashboard on first load
        if st.session_state.dashboard_data is None or refresh_button:
            with st.spinner(f"{'Initializing' if st.session_state.dashboard_data is None else 'Updating'} {asset_type} data..."):
                st.session_state.dashboard_data = update_dashboard_data(assets)
        
        # Display dashboard if we have data
        if st.session_state.dashboard_data:
            # Create DataFrame
            df = pd.DataFrame(st.session_state.dashboard_data)
            
            if not df.empty:
                # Convert Daily Change to numeric for sorting
                df['Daily Change Numeric'] = df['Daily Change'].str.rstrip('%').astype(float)
                df = df.sort_values('Daily Change Numeric', ascending=False)
                df = df.drop('Daily Change Numeric', axis=1)
                
                # Order columns
                columns = ['Stock', 'Current Price', 'Daily Change', 
                         'Buy Build Up', 'Buy Run Up', 'Sell Build Up', 'Sell Run Up']
                df = df[columns]
                
                # Display summary section
                st.markdown("#### Phase Summary")
                summary_df = create_summary_section(df)
                st.dataframe(summary_df, use_container_width=True)
                
                # Add some space between summary and main table
                st.markdown("---")
                st.markdown("#### Stock Details")
                
                #Style the dataframe
                def style_phases(x):
                    styles = pd.Series([''] * len(x), index=x.index)
                    
                    # Style for Build Up phases (Setup)
                    if x.name in ['Buy Build Up', 'Sell Build Up']:
                        # Add background color for 9s
                        styles[x == '9'] = 'font-weight: 900; color: black; font-size: 20px; background-color: #90EE90'  # Light green
                        # Normal styling for other digits
                        styles[x.astype(str).str.isdigit() & (x != '9')] = 'color: black; font-weight: bold'
                    
                    # Style for Run Up phases (Countdown)
                    if x.name in ['Buy Run Up', 'Sell Run Up']:
                        # Add background color for 13s
                        styles[x == '13'] = 'font-weight: 900; color: red; font-size: 20px; background-color: #FFB6C1'  # Light pink
                        # Normal styling for other digits
                        styles[x.astype(str).str.isdigit() & (x != '13')] = 'color: black; font-weight: bold'
                    
                    return styles
                
                # Apply styling and display
                styled_df = df.style.apply(style_phases)
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=(len(df) + 1) * 40
                )
                
                # Generate CSV for download button
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Dashboard Data",
                    data=csv_data,
                    file_name=f"{asset_type.lower().replace(' ', '_')}_phases.csv",
                    mime="text/csv",
                    key='download_button'
                )
            else:
                st.warning(f"No data available for {asset_type}. Please refresh.")
    else:
        st.error(f"Could not fetch {asset_type} list from GitHub")

if __name__ == "__main__":
    main()
