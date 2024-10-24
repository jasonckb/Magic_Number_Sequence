import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page setup
st.set_page_config(layout="wide")
st.title("Magic Number Sequence by Jason Chan")

# Sidebar
ticker = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
if ticker.isdigit():
    ticker = f"{int(ticker):04d}.HK"

# Download data
end = datetime.now()
start = end - timedelta(days=365)
df = yf.download(ticker, start=start, end=end)

if not df.empty:
    # Create figure
    fig = go.Figure()
    
    # Add candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        )
    )
    
    # Update layout
    fig.update_layout(
        yaxis_title='Price',
        xaxis_title='Date',
        yaxis=dict(side='right'),
        height=800,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Display chart
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("No data available")
