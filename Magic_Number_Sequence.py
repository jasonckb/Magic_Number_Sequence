import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import numpy as np

# Set page configuration
st.set_page_config(layout="wide")

# Sidebar for input
ticker_inputo = st.sidebar.text_input("Enter ticker", "AAPL")

# Function to check if the ticker is formatted correctly
def check_ticker_format(ticker):
    if ticker.isdigit():
        return ticker.zfill(4) + ".HK"
    return ticker.upper()

# Function to clean Yahoo Finance data
def clean_yahoo_data(df):
    try:
        # Convert the DataFrame to numeric, coerce errors to NaN
        df_cleaned = df.apply(pd.to_numeric, errors='coerce')
        
        # Drop any rows where all OHLC values are NaN
        df_cleaned = df_cleaned.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        # Preserve the datetime index
        df_cleaned.index = pd.to_datetime(df.index)
        
        return df_cleaned
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
    
    # Create ticker object
    ticker = yf.Ticker(formatted_ticker)
    
    # Get historical data
    data = ticker.history(start=start_date, end=end_date)
    
    # Clean the data
    if not data.empty:
        data = clean_yahoo_data(data)
    
    # Check if data is empty or has invalid values
    if data.empty or len(data) == 0:
        st.error(f"No valid data available for {formatted_ticker}")
    else:
        # Title and layout
        st.title("Candlestick Chart for " + formatted_ticker)

        # Function to create TD Sequential Chart
        def create_td_sequential_chart(df):
            # Create figure
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
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
