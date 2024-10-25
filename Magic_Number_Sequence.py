import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

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
st.set_page_config(layout="wide")
st.title("Magic Number Sequence by Jason Chan")

# Function to create TD Sequential Chart
def create_td_sequential_chart(df, start_date, end_date):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'])
    ])
    fig.update_xaxes(type='category')
    st.plotly_chart(fig)

# Plot the chart
create_td_sequential_chart(data, start_date, end_date)
```

This code sets up a Streamlit app with the specifications you provided. It allows input of a ticker symbol, defaults to AAPL, formats HK stock tickers, downloads one year of historical data, and plots an interactive candlestick chart using Plotly, excluding non-trading dates on the x-axis. The layout is set to wide, and the chart title is "Magic Number Sequence by Jason Chan."

Sources:
[1]  (https://github.com/juanlazarde/fybot/tree/06bd5a8b175255a57a9867d6cbe4f3ba18715d98/fybot%2Fcore%2Fchart.py)
