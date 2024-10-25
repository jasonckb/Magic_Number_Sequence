import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go

# Set page configuration
st.set_page_config(layout="wide")

# Sidebar for input
ticker_input = st.sidebar.text_input("Enter ticker", "AAPL")

# Format ticker for HK stocks
if ticker_input.isdigit():
    ticker_input = ticker_input.zfill(4) + ".HK"

# Download 1 year of data
end_date = pd.Timestamp.today()
start_date = end_date - pd.DateOffset(years=1)
data = yf.download(ticker_input, start=start_date, end=end_date)

# Title and layout
st.title("Candlestick Chart for " + ticker_input)

# Function to create TD Sequential Chart
def create_td_sequential_chart(df):
    if df.empty:
        st.write("No data available for the selected ticker.")
        return
    
    # Ensure index is datetime
    df.index = pd.to_datetime(df.index)

    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'])
    ])
    
    fig.update_xaxes(type='category')
    fig.update_layout(title='Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
    
    st.plotly_chart(fig)

# Call the function to create the chart
create_td_sequential_chart(data)
