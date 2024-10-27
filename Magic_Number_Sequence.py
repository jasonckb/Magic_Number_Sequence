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
        
def check_bar8_rule(df, current_idx, bar8_idx, is_buy_countdown):
    """
    Helper function to check if bar 8 rule is met
    """
    if is_buy_countdown:
        return safe_compare(df['Low'].iloc[current_idx], df['Close'].iloc[bar8_idx], '<=')
    else:
        return safe_compare(df['High'].iloc[current_idx], df['Close'].iloc[bar8_idx], '>=')
        
def check_recycle_completion(current_idx, start_idx, setup_values):
    """Check if a recycle condition has completed within 18 bars"""
    if start_idx < 0 or current_idx - start_idx > 18:
        return False
    # Find the maximum setup value in the range
    max_setup = max(setup_values[start_idx:current_idx + 1])
    return max_setup == 9

def calculate_td_sequential(df):
    """
    Calculate TD Sequential indicators including TDST and additional checks
    """
    # Initialize columns
    df['buy_setup'] = 0
    df['sell_setup'] = 0
    df['buy_countdown'] = 0
    df['sell_countdown'] = 0
    df['setup_phase'] = ''
    df['tdst_support'] = 0
    df['tdst_resistance'] = 0
    df['buy_setup_perfection'] = False
    df['sell_setup_perfection'] = False
    df['plus_count'] = 0
    
    # Minimum bars needed for calculation
    min_bars = 4
    
    for i in range(min_bars, len(df)):
        # Price flip conditions using safe comparison
        buy_flip = safe_compare(df['Close'].iloc[i], df['Close'].iloc[i-4], '<') and \
                   safe_compare(df['Close'].iloc[i-1], df['Close'].iloc[i-5], '>')
        sell_flip = safe_compare(df['Close'].iloc[i], df['Close'].iloc[i-4], '>') and \
                    safe_compare(df['Close'].iloc[i-1], df['Close'].iloc[i-5], '<')
        
        # Setup Phase Logic with TDST
        if buy_flip:
            if df['sell_setup'].iloc[i-1] > 0 and df['sell_setup'].iloc[i-1] < 9:
                # Check for consecutive sequence using safe comparison
                consecutive = True
                for j in range(i-3, i+1):
                    if not safe_compare(df['Close'].iloc[j], df['Close'].iloc[j-4], '<'):
                        consecutive = False
                        break
                if consecutive:
                    df.loc[i, 'sell_setup'] = df['sell_setup'].iloc[i-1] + 1
                else:
                    df.loc[i, 'sell_setup'] = 0
            else:
                df.loc[i, 'sell_setup'] = 1
                
        if sell_flip:
            if df['buy_setup'].iloc[i-1] > 0 and df['buy_setup'].iloc[i-1] < 9:
                # Check for consecutive sequence using safe comparison
                consecutive = True
                for j in range(i-3, i+1):
                    if not safe_compare(df['Close'].iloc[j], df['Close'].iloc[j-4], '>'):
                        consecutive = False
                        break
                if consecutive:
                    df.loc[i, 'buy_setup'] = df['buy_setup'].iloc[i-1] + 1
                else:
                    df.loc[i, 'buy_setup'] = 0
            else:
                df.loc[i, 'buy_setup'] = 1
        
        # TDST Support/Resistance using safe_minmax
        if df['buy_setup'].iloc[i-1] == 9:
            df.loc[i, 'tdst_support'] = safe_minmax(df['Low'].iloc[i-8:i+1], 'min')
        if df['sell_setup'].iloc[i-1] == 9:
            df.loc[i, 'tdst_resistance'] = safe_minmax(df['High'].iloc[i-8:i+1], 'max')
            
        # Setup Perfection using safe_minmax
        if df['buy_setup'].iloc[i] == 9:
            low_6_7 = safe_minmax([df['Low'].iloc[i-2], df['Low'].iloc[i-3]], 'min')
            low_8_9 = safe_minmax([df['Low'].iloc[i], df['Low'].iloc[i-1]], 'min')
            df.loc[i, 'buy_setup_perfection'] = safe_compare(low_8_9, low_6_7, '<')
            
        if df['sell_setup'].iloc[i] == 9:
            high_6_7 = safe_minmax([df['High'].iloc[i-2], df['High'].iloc[i-3]], 'max')
            high_8_9 = safe_minmax([df['High'].iloc[i], df['High'].iloc[i-1]], 'max')
            df.loc[i, 'sell_setup_perfection'] = safe_compare(high_8_9, high_6_7, '>')
        
        # Clear interrupted setups after 4 bars
        if df['buy_setup'].iloc[i-1] > 0 and df['buy_setup'].iloc[i-1] < 9:
            if not buy_flip and i - df['buy_setup'].iloc[i-1] >= 4:
                df.loc[i, 'buy_setup'] = 0
                
        if df['sell_setup'].iloc[i-1] > 0 and df['sell_setup'].iloc[i-1] < 9:
            if not sell_flip and i - df['sell_setup'].iloc[i-1] >= 4:
                df.loc[i, 'sell_setup'] = 0
        
        # Countdown Phase Logic with Plus Count
        if df['buy_setup'].iloc[i-1] == 9:
            df.loc[i, 'buy_countdown'] = 1
            df.loc[i, 'setup_phase'] = 'buy_countdown'
            df.loc[i, 'plus_count'] = 0
            
        elif df['sell_setup'].iloc[i-1] == 9:
            df.loc[i, 'sell_countdown'] = 1
            df.loc[i, 'setup_phase'] = 'sell_countdown'
            df.loc[i, 'plus_count'] = 0
            
        # Continue countdown if in progress with bar 8 rule check
        if df['setup_phase'].iloc[i-1] == 'buy_countdown':
            if safe_compare(df['Close'].iloc[i], df['Low'].iloc[i-2], '<'):
                df.loc[i, 'buy_countdown'] = df['buy_countdown'].iloc[i-1] + 1
                df.loc[i, 'setup_phase'] = 'buy_countdown'
                
                # Check for + after bar 12
                if df['buy_countdown'].iloc[i] >= 12:
                    if check_bar8_rule(df, i, i-8, True):
                        df.loc[i, 'plus_count'] = df['plus_count'].iloc[i-1] + 1
                
            # Allow new sell setup during buy countdown
            if sell_flip:
                df.loc[i, 'sell_setup'] = 1
                
        elif df['setup_phase'].iloc[i-1] == 'sell_countdown':
            if safe_compare(df['Close'].iloc[i], df['High'].iloc[i-2], '>'):
                df.loc[i, 'sell_countdown'] = df['sell_countdown'].iloc[i-1] + 1
                df.loc[i, 'setup_phase'] = 'sell_countdown'
                
                # Check for + after bar 12
                if df['sell_countdown'].iloc[i] >= 12:
                    if check_bar8_rule(df, i, i-8, False):
                        df.loc[i, 'plus_count'] = df['plus_count'].iloc[i-1] + 1
                
            # Allow new buy setup during sell countdown
            if buy_flip:
                df.loc[i, 'buy_setup'] = 1
        
        # Reset countdown if conditions are not met
        if df['buy_countdown'].iloc[i-1] > 0:
            if not safe_compare(df['Close'].iloc[i], df['Low'].iloc[i-2], '<'):
                df.loc[i, 'buy_countdown'] = 0
                df.loc[i, 'setup_phase'] = ''
                df.loc[i, 'plus_count'] = 0
                
        if df['sell_countdown'].iloc[i-1] > 0:
            if not safe_compare(df['Close'].iloc[i], df['High'].iloc[i-2], '>'):
                df.loc[i, 'sell_countdown'] = 0
                df.loc[i, 'setup_phase'] = ''
                df.loc[i, 'plus_count'] = 0
                
        # TDST Break Rules with safe comparison
        if df['tdst_support'].iloc[i-1] > 0:
            if safe_compare(df['Close'].iloc[i], df['tdst_support'].iloc[i-1], '<'):
                df.loc[i, 'tdst_support'] = 0
                
        if df['tdst_resistance'].iloc[i-1] > 0:
            if safe_compare(df['Close'].iloc[i], df['tdst_resistance'].iloc[i-1], '>'):
                df.loc[i, 'tdst_resistance'] = 0
    
    return df


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


