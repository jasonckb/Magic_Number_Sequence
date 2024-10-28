import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import numpy as np
import requests
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(layout="wide")
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

def get_current_phase(df):
    """Get the current phase numbers for a stock"""
    buy_setup, sell_setup, buy_countdown, sell_countdown, _, _, _, _, _ = calculate_td_sequential(df)
    
    # Get the last non-zero values
    current_phases = {
        'Buy Build Up': '-',
        'Sell Build Up': '-',
        'Buy Run Up': '-',
        'Sell Run Up': '-'
    }
    
    # Get the most recent non-zero values for each phase
    if len(buy_setup) > 0:
        last_buy_setup = next((str(int(x)) for x in reversed(buy_setup) if x > 0), '-')
        current_phases['Buy Build Up'] = last_buy_setup
        
    if len(sell_setup) > 0:
        last_sell_setup = next((str(int(x)) for x in reversed(sell_setup) if x > 0), '-')
        current_phases['Sell Build Up'] = last_sell_setup
        
    if len(buy_countdown) > 0:
        last_buy_countdown = next((str(int(x)) for x in reversed(buy_countdown) if x > 0), '-')
        current_phases['Buy Run Up'] = last_buy_countdown
        
    if len(sell_countdown) > 0:
        last_sell_countdown = next((str(int(x)) for x in reversed(sell_countdown) if x > 0), '-')
        current_phases['Sell Run Up'] = last_sell_countdown
    
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

def check_ticker_format(ticker):
    if ticker.isdigit():
        return ticker.zfill(4) + ".HK"
    return ticker.upper()

def clean_yahoo_data(df):
    try:
        # Handle multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            # Get the first level of column names and the ticker
            base_cols = df.columns.get_level_values(0).unique()
            ticker = df.columns.get_level_values(1)[0]  # Get ticker from any column
            
            # Create a new dataframe with single-level columns
            df_cleaned = pd.DataFrame(index=df.index)
            for col in base_cols:
                df_cleaned[col] = df[(col, ticker)]
        else:
            df_cleaned = df.copy()
        
        # Drop any rows with missing values
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

def check_bar8_rule(df, current_idx, bar8_idx, is_buy_countdown):
    """Helper function to check if bar 8 rule is met"""
    if is_buy_countdown:
        return safe_compare(df['Low'].iloc[current_idx], df['Close'].iloc[bar8_idx], '<=')
    else:
        return safe_compare(df['High'].iloc[current_idx], df['Close'].iloc[bar8_idx], '>=')

def check_18_bar_rule(df, current_idx, last_flip_idx, is_buy_countdown):
    """
    Helper function to check 18-bar rule:
    - For buy countdown: checks if no buy flip in last 18 bars after sell flip
    - For sell countdown: checks if no sell flip in last 18 bars after buy flip
    Returns True if countdown should be cancelled
    """
    if last_flip_idx < 0 or current_idx - last_flip_idx < 18:
        return False
        
    # Check if no same-direction flip in last 18 bars
    for i in range(last_flip_idx, current_idx + 1):
        if is_buy_countdown and check_buy_flip(df, i):
            return False
        elif not is_buy_countdown and check_sell_flip(df, i):
            return False
    return True

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
        
        # Track if setup 1 occurs at this bar
        setup_one_at_current_bar = False
        
        # Track if setup 1 occurs at this bar
        setup_one_at_current_bar = False
        
        # Buy Setup Phase
        if check_buy_flip(df, i):
            # Allow setup 1 if:
            # - No plus seen, OR
            # - This bar has buy plus or buy 13
            if (not buy_plus_without_setup or 
                (buy_deferred[i] or  # Has plus on this bar
                 (waiting_for_buy_13 and safe_compare(df['Low'].iloc[i], bar8_close_buy, '<=')))):  # Has bar 13 on this bar
                buy_setup_active = True
                sell_setup_active = False
                setup_start_idx = i
                buy_setup[i] = 1
                setup_one_at_current_bar = True
        elif buy_setup_active:
            # Check if setup should continue
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
                # Clear interrupted setup
                buy_setup_active = False
                # Only clear if we've had 4 bars without continuation
                if i - setup_start_idx >= 4:
                    buy_setup[setup_start_idx:i+1] = 0
        
        # Sell Setup Phase
        if check_sell_flip(df, i):
            # Allow setup 1 if:
            # - No plus seen, OR
            # - This bar has sell plus or sell 13
            if (not sell_plus_without_setup or 
                (sell_deferred[i] or  # Has plus on this bar
                 (waiting_for_sell_13 and safe_compare(df['High'].iloc[i], bar8_close_sell, '>=')))):  # Has bar 13 on this bar
                sell_setup_active = True
                buy_setup_active = False
                setup_start_idx = i
                sell_setup[i] = 1
                setup_one_at_current_bar = True
        elif sell_setup_active:
            # Check if setup should continue
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
                # Clear interrupted setup
                sell_setup_active = False
                # Only clear if we've had 4 bars without continuation
                if i - setup_start_idx >= 4:
                    sell_setup[setup_start_idx:i+1] = 0
        
        # Buy Countdown Phase - Only if no sell countdown active
        if not sell_countdown_active:
            if buy_setup_complete and not buy_countdown_active and not need_new_buy_setup:
                if safe_compare(df['Close'].iloc[i], df['Low'].iloc[i-2], '<='):
                    buy_countdown_active = True
                    buy_setup_complete = False
                    buy_countdown[i] = 1
                    buy_setup_count = 1
                    buy_countdown_bars = [i]
                    waiting_for_buy_13 = False
                    bar8_close_buy = None
                    
            elif buy_countdown_active:
                if waiting_for_buy_13:
                    # For bar 13, only check bar 8 rule
                    if safe_compare(df['Low'].iloc[i], bar8_close_buy, '<='):
                        buy_countdown[i] = 13
                        buy_countdown_active = False
                        waiting_for_buy_13 = False
                        bar8_close_buy = None
                        need_new_buy_setup = True
                        buy_plus_without_setup = False  # Reset plus prevention on bar 13
                    else:
                        # For '+', still need 2-bar rule
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
        
        # Sell Countdown Phase - Only if no buy countdown active
        if not buy_countdown_active:
            if sell_setup_complete and not sell_countdown_active and not need_new_sell_setup:
                if safe_compare(df['Close'].iloc[i], df['High'].iloc[i-2], '>='):
                    sell_countdown_active = True
                    sell_setup_complete = False
                    sell_countdown[i] = 1
                    sell_setup_count = 1
                    sell_countdown_bars = [i]
                    waiting_for_sell_13 = False
                    bar8_close_sell = None
                    
            elif sell_countdown_active:
                if waiting_for_sell_13:
                    # For bar 13, only check bar 8 rule
                    if safe_compare(df['High'].iloc[i], bar8_close_sell, '>='):
                        sell_countdown[i] = 13
                        sell_countdown_active = False
                        waiting_for_sell_13 = False
                        bar8_close_sell = None
                        need_new_sell_setup = True
                        sell_plus_without_setup = False  # Reset plus prevention on bar 13
                    else:
                        # For '+', still need 2-bar rule
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
    # Remove the st.set_page_config from here
    
    # Create two columns for layout
    left_col, right_col = st.columns([1, 3])
    
    # PART 1: Single Stock Chart Analysis (Left Column)
    with left_col:
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
    
    # PART 2: Dashboard of HK Stocks (Right Column)
    with right_col:
        st.markdown("### HK Stocks Dashboard")
        
        # Fetch stocks from GitHub
        hk_stocks = get_stocks_from_github()
        
        if hk_stocks:
            if st.button("Refresh Dashboard"):
                with st.spinner("Updating dashboard data..."):
                    dashboard_data = update_dashboard_data(hk_stocks)
                    
                    if dashboard_data:
                        # Create DataFrame
                        df = pd.DataFrame(dashboard_data)
                        
                        # Style the dataframe
                        def highlight_phases(val):
                            if isinstance(val, str) and val.isdigit():
                                return 'background-color: #90EE90' if int(val) > 0 else ''
                            return ''
                        
                        # Apply styling
                        styled_df = df.style.apply(lambda x: [highlight_phases(v) for v in x])
                        
                        # Display the dashboard
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # Option to download as CSV
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Dashboard Data",
                            data=csv,
                            file_name="stock_phases.csv",
                            mime="text/csv"
                        )
        else:
            st.error("Could not fetch HK stocks list from GitHub")

if __name__ == "__main__":
    main()
