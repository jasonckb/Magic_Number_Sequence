import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import numpy as np

# Set page configuration
st.set_page_config(layout="wide")

# Sidebar for input
ticker_input = st.sidebar.text_input("Enter ticker", "AAPL")

# Function to check if the ticker is formatted correctly
def check_ticker_format(ticker):
    if ticker.isdigit():
        return ticker.zfill(4) + ".HK"
    return ticker.upper()

# Function to clean Yahoo Finance data
def clean_yahoo_data(df):
    try:
        # Make a copy to avoid modifying original data
        df = df.copy()
        
        # Ensure all OHLC columns are float type
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop any rows with NaN values
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        # Reset index if needed while keeping dates
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        return df
    except Exception as e:
        st.error(f"Error in data cleaning: {str(e)}")
        return pd.DataFrame()

# Format the ticker input
formatted_ticker = check_ticker_format(ticker_input)
st.write("Formatted Ticker: ", formatted_ticker)

try:
    # Download 1 year of data
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=1)
    data = yf.download(formatted_ticker, start=start_date, end=end_date, progress=False)
    
    # Clean the data
    data = clean_yahoo_data(data)
    
    # Check if data is empty or has invalid values
    if data.empty or len(data) == 0:
        st.error(f"No valid data available for {formatted_ticker}")
    else:
        # Title and layout
        st.title("Candlestick Chart for " + formatted_ticker)

        # Function to create TD Sequential Chart
        def create_td_sequential_chart(df):
            # Ensure data types are correct
            x_dates = df.index.astype(str).tolist()
            
            # Create figure
            fig = go.Figure(data=[go.Candlestick(
                x=x_dates,
                open=df['Open'].tolist(),
                high=df['High'].tolist(),
                low=df['Low'].tolist(),
                close=df['Close'].tolist(),
                name=formatted_ticker
            )])
            
            # Update layout
            fig.update_layout(
                title={
                    'text': 'Candlestick Chart',
                    'y': 0.9,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title='Date',
                yaxis_title='Price',
                template='plotly_white',
                xaxis_rangeslider_visible=False
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)

        # Call the function to create the chart
        create_td_sequential_chart(data)
        
        # Display recent data
        st.write("Recent Data:")
        st.dataframe(data.tail())

except Exception as e:
    st.error(f"Error occurred: {str(e)}")
