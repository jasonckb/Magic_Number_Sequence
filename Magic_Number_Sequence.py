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

# Sidebar input
ticker_input = st.sidebar.text_input("Enter ticker", "AAPL")

def get_stocks_from_github():
    """Fetch stock list from GitHub"""
    try:
        github_url = "https://raw.githubusercontent.com/jasonckb/Magic_Number_Sequence/main/HK%20Stocks.txt"
        response = requests.get(github_url)
        response.raise_for_status()
        stocks = [line.strip() for line in response.text.splitlines() if line.strip()]
        return stocks
    except Exception as e:
        st.error(f"Error fetching stocks from GitHub: {str(e)}")
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

# Comparison Helper Functions
def safe_compare(a, b, operator='<='):
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

def get_tdst_level(df, setup_start_idx, end_idx, is_buy_setup):
    if is_buy_setup:
        prior_close = float(df['Close'].iloc[setup_start_idx-1]) if setup_start_idx > 0 else float('-inf')
        highest_high = safe_minmax(df['High'].iloc[setup_start_idx:end_idx+1], 'max')
        return safe_minmax([prior_close, highest_high], 'max')
    else:
        prior_close = float(df['Close'].iloc[setup_start_idx-1]) if setup_start_idx > 0 else float('inf')
        lowest_low = safe_minmax(df['Low'].iloc[setup_start_idx:end_idx+1], 'min')
        return safe_minmax([prior_close, lowest_low], 'min')

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

def calculate_td_sequential(df):
    # Initialize arrays
    buy_setup = np.zeros(len(df))
    sell_setup = np.zeros(len(df))
    buy_countdown = np.zeros(len(df))
    sell_countdown = np.zeros(len(df))
    buy_perfection = np.zeros(len(df))
    sell_perfection = np.zeros(len(df))
    buy_deferred = np.zeros(len(df), dtype=bool)
    sell_deferred = np.zeros(len(df), dtype=bool)
    
    # Initialize state variables
    buy_setup_active = False
    sell_setup_active = False
    buy_countdown_active = False
    sell_countdown_active = False
    buy_setup_complete = False
    sell_setup_complete = False
    setup_start_idx = 0
    
    # Initialize plus prevention flags
    buy_plus_without_setup = False
    sell_plus_without_setup = False
    
    # Initialize countdown tracking
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
    
    # Initialize 18-bar rule tracking
    last_opposite_flip = {'buy': -1, 'sell': -1}
    
    for i in range(len(df)):
        # Check TDST violations
        if buy_countdown_active and tdst.check_resistance_violation(df['Close'].iloc[i]):
            buy_countdown_active = False
            buy_setup_count = 0
            buy_countdown_bars = []
            waiting_for_buy_13 = False
            bar8_close_buy = None
            buy_plus_without_setup = False
            
        if sell_countdown_active and tdst.check_support_violation(df['Close'].iloc[i]):
            sell_countdown_active = False
            sell_setup_count = 0
            sell_countdown_bars = []
            waiting_for_sell_13 = False
            bar8_close_sell = None
            sell_plus_without_setup = False
        
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
            buy_plus_without_setup = False
            
        if sell_countdown_active and check_18_bar_rule(df, i, last_opposite_flip['sell'], False):
            sell_countdown_active = False
            sell_setup_count = 0
            sell_countdown_bars = []
            waiting_for_sell_13 = False
            bar8_close_sell = None
            sell_plus_without_setup = False
        
        # Track if setup completes on this bar
        buy_setup_completed_this_bar = False
        sell_setup_completed_this_bar = False
        setup_one_at_current_bar = False
        
        # Buy Setup Phase
        if check_buy_flip(df, i):  # Valid buy price flip
            if (not buy_plus_without_setup or 
                (buy_deferred[i] or  
                 (waiting_for_buy_13 and safe_compare(df['Low'].iloc[i], bar8_close_buy, '<=')))):
                buy_setup_active = True
                sell_setup_active = False  # Cancel any active sell setup
                setup_start_idx = i
                buy_setup[i] = 1
                setup_one_at_current_bar = True
                # Clear any existing sell setup counts
                if i > 0 and sell_setup[i-1] > 0:
                    sell_setup[i-1] = 0
        elif buy_setup_active:
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
                            buy_setup_completed_this_bar = True
                            need_new_buy_setup = False
                            resistance = get_tdst_level(df, setup_start_idx, i, True)
                            tdst.add_resistance(resistance, df.index[i])
                else:
                    buy_setup[i] = 1
            else:
                buy_setup_active = False
                if i - setup_start_idx >= 4:
                    buy_setup[setup_start_idx:i+1] = 0
        
        # Sell Setup Phase
        if check_sell_flip(df, i):  # Valid sell price flip
            if (not sell_plus_without_setup or 
                (sell_deferred[i] or  
                 (waiting_for_sell_13 and safe_compare(df['High'].iloc[i], bar8_close_sell, '>=')))):
                sell_setup_active = True
                buy_setup_active = False  # Cancel any active buy setup
                setup_start_idx = i
                sell_setup[i] = 1
                setup_one_at_current_bar = True
                # Clear any existing buy setup counts
                if i > 0 and buy_setup[i-1] > 0:
                    buy_setup[i-1] = 0
        elif sell_setup_active:
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
                            sell_setup_completed_this_bar = True
                            need_new_sell_setup = False
                            support = get_tdst_level(df, setup_start_idx, i, False)
                            tdst.add_support(support, df.index[i])
                else:
                    sell_setup[i] = 1
            else:
                sell_setup_active = False
                if i - setup_start_idx >= 4:
                    sell_setup[setup_start_idx:i+1] = 0
        
        # Buy Countdown Phase
        if not sell_countdown_active:
            if (buy_setup_completed_this_bar or buy_setup_complete) and not buy_countdown_active and not need_new_buy_setup:
                if i >= 2 and safe_compare(df['Close'].iloc[i], df['Low'].iloc[i-2], '<='):
                    buy_countdown_active = True
                    buy_setup_complete = False
                    buy_countdown[i] = 1
                    buy_setup_count = 1
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
                        need_new_buy_setup = True
                        buy_plus_without_setup = False
                    else:
                        if safe_compare(df['Close'].iloc[i], df['Low'].iloc[i-2], '<='):
                            buy_deferred[i] = True
                            if not setup_one_at_current_bar:
                                buy_plus_without_setup = True
                else:
                    if safe_compare(df['Close'].iloc[i], df['Low'].iloc[i-2], '<='):
                        buy_countdown_bars.append(i)
                        if buy_setup_count < 12:
                            buy_setup_count += 1
                            buy_countdown[i] = buy_setup_count
                            if buy_setup_count == 8:
                                bar8_close_buy = float(df['Close'].iloc[i])
                            elif buy_setup_count == 12:
                                waiting_for_buy_13 = True
        
        # Sell Countdown Phase
        if not buy_countdown_active:
            if (sell_setup_completed_this_bar or sell_setup_complete) and not sell_countdown_active and not need_new_sell_setup:
                if i >= 2 and safe_compare(df['Close'].iloc[i], df['High'].iloc[i-2], '>='):
                    sell_countdown_active = True
                    sell_setup_complete = False
                    sell_countdown[i] = 1
                    sell_setup_count = 1
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
                        need_new_sell_setup = True
                        sell_plus_without_setup = False
                    else:
                        if safe_compare(df['Close'].iloc[i], df['High'].iloc[i-2], '>='):
                            sell_deferred[i] = True
                            if not setup_one_at_current_bar:
                                sell_plus_without_setup = True
                else:
                    if safe_compare(df['Close'].iloc[i], df['High'].iloc[i-2], '>='):
                        sell_countdown_bars.append(i)
                        if sell_setup_count < 12:
                            sell_setup_count += 1
                            sell_countdown[i] = sell_setup_count
                            if sell_setup_count == 8:
                                bar8_close_sell = float(df['Close'].iloc[i])
                            elif sell_setup_count == 12:
                                waiting_for_sell_13 = True
    
    return buy_setup, sell_setup, buy_countdown, sell_countdown, buy_perfection, sell_perfection, buy_deferred, sell_deferred, tdst

def create_td_sequential_chart(df, ticker):
    if df.empty:
        return None
        
    # Calculate TD Sequential indicators
    buy_setup, sell_setup, buy_countdown, sell_countdown, buy_perfection, \
    sell_perfection, buy_deferred, sell_deferred, tdst = calculate_td_sequential(df)
    
    # Create base figure with candlesticks
    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC',
            increasing_line_color='#26A69A',
            decreasing_line_color='#EF5350'
        )
    ])
    
    # Calculate y-axis range
    y_min = float(df['Low'].min())
    y_max = float(df['High'].max())
    y_padding = (y_max - y_min) * 0.1
    
    # Add TrendLine levels
    for i, (level, date) in enumerate(tdst.resistance_levels):
        if date in df.index:
            date_idx = df.index.get_loc(date)
            end_idx = min(date_idx + 30, len(df.index) - 1)
            end_date = df.index[end_idx]
            
            # Add horizontal line
            fig.add_shape(
                type="line",
                x0=date,
                x1=end_date,
                y0=float(level),
                y1=float(level),
                line=dict(color="red", width=1, dash="dash"),
                layer="below"
            )
            
            # Add label
            fig.add_annotation(
                x=end_date,
                y=float(level),
                text=f"TrendLine R{len(tdst.resistance_levels)-i}",
                showarrow=False,
                xanchor="left",
                xshift=10,
                font=dict(color="red", size=10)
            )
    
    for i, (level, date) in enumerate(tdst.support_levels):
        if date in df.index:
            date_idx = df.index.get_loc(date)
            end_idx = min(date_idx + 30, len(df.index) - 1)
            end_date = df.index[end_idx]
            
            # Add horizontal line
            fig.add_shape(
                type="line",
                x0=date,
                x1=end_date,
                y0=float(level),
                y1=float(level),
                line=dict(color="green", width=1, dash="dash"),
                layer="below"
            )
            
            # Add label
            fig.add_annotation(
                x=end_date,
                y=float(level),
                text=f"TrendLine S{len(tdst.support_levels)-i}",
                showarrow=False,
                xanchor="left",
                xshift=10,
                font=dict(color="green", size=10)
            )
    
    # Add setup and countdown annotations
    for i in range(len(df)):
        # Buy setup counts
        if buy_setup[i] > 0:
            count_text = str(int(buy_setup[i]))
            font_size = 13 if buy_setup[i] == 9 else 10
            if buy_setup[i] == 9:
                count_text += "↑" if buy_perfection[i] else "+"
            fig.add_annotation(
                x=df.index[i],
                y=float(df['Low'].iloc[i]),
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
                x=df.index[i],
                y=float(df['Low'].iloc[i]),
                text=count_text,
                showarrow=False,
                yshift=-25,
                font=dict(color="red", size=font_size)
            )
        
        # Sell setup counts
        if sell_setup[i] > 0:
            count_text = str(int(sell_setup[i]))
            font_size = 13 if sell_setup[i] == 9 else 10
            if sell_setup[i] == 9:
                count_text += "↓" if sell_perfection[i] else "+"
            fig.add_annotation(
                x=df.index[i],
                y=float(df['High'].iloc[i]),
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
                x=df.index[i],
                y=float(df['High'].iloc[i]),
                text=count_text,
                showarrow=False,
                yshift=25,
                font=dict(color="red", size=font_size)
            )
    
    # Update layout
    fig.update_layout(
        title=f'Magic Number Sequence Analysis - {ticker}',
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

def main():
    # Initialize session state if not exists
    if 'dashboard_data' not in st.session_state:
        st.session_state.dashboard_data = None
    if 'last_ticker' not in st.session_state:
        st.session_state.last_ticker = None
    
    # PART 1: Single Stock Chart Analysis
    st.markdown("### Single Stock Analysis")
    
    # Only refresh chart if ticker changed
    if ticker_input != st.session_state.last_ticker:
        st.session_state.last_ticker = ticker_input
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
                            
                            # Store the chart data in session state
                            st.session_state.last_chart_data = data
                            st.session_state.last_chart_fig = fig
            except Exception as e:
                st.error(f"Error analyzing {ticker_input}: {str(e)}")
    else:
        # Re-display existing chart without recalculation
        if ticker_input:
            formatted_ticker = check_ticker_format(ticker_input)
            st.write(f"Analyzing: {formatted_ticker}")
            
            # Check if we have stored chart data
            if hasattr(st.session_state, 'last_chart_fig') and st.session_state.last_chart_fig is not None:
                st.plotly_chart(st.session_state.last_chart_fig, use_container_width=True)
            else:
                # Regenerate if not stored
                try:
                    end_date = pd.Timestamp.today()
                    start_date = end_date - pd.DateOffset(years=1)
                    data = yf.download(formatted_ticker, start=start_date, end=end_date, progress=False)
                    
                    if not data.empty:
                        data = clean_yahoo_data(data)
                        if len(data) > 0:
                            fig = create_td_sequential_chart(data, formatted_ticker)
                            if fig is not None:
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Store the chart data
                                st.session_state.last_chart_data = data
                                st.session_state.last_chart_fig = fig
                except Exception as e:
                    st.error(f"Error analyzing {ticker_input}: {str(e)}")
    
    # PART 2: Dashboard Controls in Sidebar and Dashboard Display
    st.sidebar.markdown("### HK Stocks Dashboard")
    
    # Fetch stocks from GitHub
    hk_stocks = get_stocks_from_github()
    
    if hk_stocks:
        # Initialize dashboard on first load
        if st.session_state.dashboard_data is None:
            with st.spinner("Initializing dashboard data..."):
                st.session_state.dashboard_data = update_dashboard_data(hk_stocks)
        
        # Refresh button for dashboard only
        if st.sidebar.button("Refresh Dashboard"):
            with st.spinner("Updating dashboard data..."):
                st.session_state.dashboard_data = update_dashboard_data(hk_stocks)
        
        # Display dashboard
        if st.session_state.dashboard_data:
            st.markdown("### HK Stocks Dashboard")
            
            # Create DataFrame
            df = pd.DataFrame(st.session_state.dashboard_data)
            
            # Order columns
            columns = ['Stock', 'Current Price', 'Daily Change', 
                      'Buy Build Up', 'Buy Run Up', 'Sell Build Up', 'Sell Run Up']
            df = df[columns]
            
            # Define the styling function with larger font sizes
            def style_phases(x):
                styles = pd.Series([''] * len(x), index=x.index)
                
                # Style for Build Up phases (9) - increased font size
                if x.name in ['Buy Build Up', 'Sell Build Up']:
                    mask = x == '9'
                    styles[mask] = 'font-weight: 900; color: green; font-size: 20px'
                
                # Style for Run Up phases (13) - increased font size
                if x.name in ['Buy Run Up', 'Sell Run Up']:
                    mask = x == '13'
                    styles[mask] = 'font-weight: 900; color:red; font-size: 20px'
                    
                return styles
            
            # Apply the styling
            styled_df = df.style.apply(style_phases)
            
            # Display the dashboard with increased row height
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
                file_name="stock_phases.csv",
                mime="text/csv",
                key='download_button'
            )
    else:
        st.sidebar.error("Could not fetch HK stocks list from GitHub")

if __name__ == "__main__":
    main()


