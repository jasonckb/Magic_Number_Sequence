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

