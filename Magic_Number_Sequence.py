import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import numpy as np
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(layout="wide")
st.title("Magic Number Sequence by Jason Chan")

# Sidebar input
ticker_input = st.sidebar.text_input("Enter ticker", "AAPL")

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

        # Make a copy of the DataFrame to avoid modifying the original
        df_cleaned = df.copy()
        
        # Convert each OHLC column to numeric individually
        for col in required_columns:
            try:
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
            except Exception as e:
                st.error(f"Error converting {col} column to numeric: {str(e)}")
                return pd.DataFrame()
        
        # Drop rows where any OHLC value is NaN
        df_cleaned = df_cleaned.dropna(subset=required_columns)
        
        # Check if we have any data left after cleaning
        if len(df_cleaned) == 0:
            st.error("No valid OHLC data remains after cleaning")
            return pd.DataFrame()
        
        # Preserve the datetime index
        df_cleaned.index = pd.to_datetime(df.index)
        
        # Debug information
        st.write(f"Original shape: {df.shape}")
        st.write(f"Cleaned shape: {df_cleaned.shape}")
        
        return df_cleaned
        
    except Exception as e:
        st.error(f"Error in data cleaning: {str(e)}")
        st.write("DataFrame info:")
        st.write(df.info())
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
    
    # Initialize recycle tracking variables
    potential_recycle_start = -1
    recycle_countdown_type = None
    
    # Add deferred state tracking
    buy_countdown_deferred = False
    sell_countdown_deferred = False
    
    # Initialize counters and flags
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
    
    def check_recycle_completion(current_idx, start_idx, setup_values):
        if start_idx < 0 or current_idx - start_idx > 18:
            return False
        max_setup = max(setup_values[start_idx:current_idx + 1])
        return max_setup == 9
    
    for i in range(len(df)):
        # Check TDST violations
        if buy_countdown_active and tdst.check_resistance_violation(df['Close'].iloc[i]):
            buy_countdown_active = False
            buy_setup_count = 0
            buy_countdown_bars = []
            waiting_for_buy_13 = False
            potential_recycle_start = -1
            recycle_countdown_type = None
            buy_countdown_deferred = False
            
        if sell_countdown_active and tdst.check_support_violation(df['Close'].iloc[i]):
            sell_countdown_active = False
            sell_setup_count = 0
            sell_countdown_bars = []
            waiting_for_sell_13 = False
            potential_recycle_start = -1
            recycle_countdown_type = None
            sell_countdown_deferred = False
        
        # Check for potential recycle starts
        if buy_countdown_active and check_sell_flip(df, i):
            potential_recycle_start = i
            recycle_countdown_type = 'buy'
        elif sell_countdown_active and check_buy_flip(df, i):
            potential_recycle_start = i
            recycle_countdown_type = 'sell'
            
        # Check for recycle completion
        if potential_recycle_start >= 0:
            if (recycle_countdown_type == 'buy' and 
                check_recycle_completion(i, potential_recycle_start, sell_setup)):
                buy_countdown_active = False
                buy_setup_count = 0
                buy_countdown_bars = []
                waiting_for_buy_13 = False
                potential_recycle_start = -1
                recycle_countdown_type = None
                buy_countdown_deferred = False
                buy_countdown[potential_recycle_start:i+1] = 0
                buy_deferred[potential_recycle_start:i+1] = False
                
            elif (recycle_countdown_type == 'sell' and 
                  check_recycle_completion(i, potential_recycle_start, buy_setup)):
                sell_countdown_active = False
                sell_setup_count = 0
                sell_countdown_bars = []
                waiting_for_sell_13 = False
                potential_recycle_start = -1
                recycle_countdown_type = None
                sell_countdown_deferred = False
                sell_countdown[potential_recycle_start:i+1] = 0
                sell_deferred[potential_recycle_start:i+1] = False
                
            elif i - potential_recycle_start > 18:
                potential_recycle_start = -1
                recycle_countdown_type = None
        
        # Setup flips
        if check_buy_flip(df, i) and not sell_countdown_active:
            buy_setup_active = True
            sell_setup_active = False
            setup_start_idx = i
            buy_setup[i] = 1
        elif check_sell_flip(df, i) and not buy_countdown_active:
            sell_setup_active = True
            buy_setup_active = False
            setup_start_idx = i
            sell_setup[i] = 1
        
        # Buy setup phase
        if buy_setup_active and not sell_countdown_active:
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
        if sell_setup_active and not buy_countdown_active:
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
        if not sell_countdown_active:
            if buy_setup_complete and not buy_countdown_active and not need_new_buy_setup:
                if safe_compare(df['Close'].iloc[i], df['Low'].iloc[i-2], '<='):
                    buy_countdown_active = True
                    buy_setup_complete = False
                    buy_countdown[i] = 1
                    buy_setup_count = 1
                    buy_countdown_bars = [i]
                    waiting_for_buy_13 = False
                    buy_countdown_deferred = False
                    
            elif buy_countdown_active:
                if waiting_for_buy_13:
                    # Always check bar 8 rule regardless of deferred state
                    if safe_compare(df['Low'].iloc[i], bar8_close_buy, '<='):
                        # Bar 8 rule is met - print 13 and end countdown
                        buy_countdown[i] = 13
                        buy_countdown_active = False
                        waiting_for_buy_13 = False
                        buy_countdown_bars = []
                        need_new_buy_setup = True
                        potential_recycle_start = -1
                        recycle_countdown_type = None
                        buy_countdown_deferred = False
                    elif safe_compare(df['Close'].iloc[i], df['Low'].iloc[i-2], '<='):
                        # Bar 8 rule not met - print + but keep checking
                        buy_deferred[i] = True
                        buy_countdown[i] = 0  # Clear any countdown number
                        buy_countdown_deferred = True
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
        if not buy_countdown_active:
            if sell_setup_complete and not sell_countdown_active and not need_new_sell_setup:
                if safe_compare(df['Close'].iloc[i], df['High'].iloc[i-2], '>='):
                    sell_countdown_active = True
                    sell_setup_complete = False
                    sell_countdown[i] = 1
                    sell_setup_count = 1
                    sell_countdown_bars = [i]
                    waiting_for_sell_13 = False
                    sell_countdown_deferred = False
                    
            elif sell_countdown_active:
                if waiting_for_sell_13:
                    # Always check bar 8 rule regardless of deferred state
                    if safe_compare(df['High'].iloc[i], bar8_close_sell, '>='):
                        # Bar 8 rule is met - print 13 and end countdown
                        sell_countdown[i] = 13
                        sell_countdown_active = False
                        waiting_for_sell_13 = False
                        sell_countdown_bars = []
                        need_new_sell_setup = True
                        potential_recycle_start = -1
                        recycle_countdown_type = None
                        sell_countdown_deferred = False
                    elif safe_compare(df['Close'].iloc[i], df['High'].iloc[i-2], '>='):
                        # Bar 8 rule not met - print + but keep checking
                        sell_deferred[i] = True
                        sell_countdown[i] = 0  # Clear any countdown number
                        sell_countdown_deferred = True
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
    try:
        # Get formatted ticker
        formatted_ticker = check_ticker_format(ticker_input)
        st.write("Formatted Ticker: ", formatted_ticker)
        
        # Download data
        end_date = pd.Timestamp.today()
        start_date = end_date - pd.DateOffset(years=1)
        
        st.write("Downloading data...")
        data = yf.download(formatted_ticker, start=start_date, end=end_date, progress=False)
        
        if not data.empty:
            # Clean and verify data
            data = clean_yahoo_data(data)
            
            if len(data) > 0:
                st.success("Data processed successfully!")
                
                # Create and display chart
                st.write("Creating chart...")
                fig = create_td_sequential_chart(data, formatted_ticker)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                    
                # Display recent data
                with st.expander("View Recent Data"):
                    st.dataframe(data.tail())
            else:
                st.error("No valid data after cleaning")
        else:
            st.error(f"No data available for {formatted_ticker}")
            
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
