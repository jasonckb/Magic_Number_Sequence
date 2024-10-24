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
    # Calculate y-axis range
    y_min = df['Low'].min()
    y_max = df['High'].max()
    y_range = y_max - y_min
    y_padding = y_range * 0.1
    
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
    
    # Update layout with fixed y-axis range
    fig.update_layout(
        yaxis=dict(
            title='Price',
            side='right',
            range=[y_min - y_padding, y_max + y_padding],  # Set fixed range with padding
            autorange=False,  # Disable autorange
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGrey'
        ),
        xaxis=dict(
            title='Date',
            rangeslider=dict(visible=False),
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGrey'
        ),
        height=800,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    # Display chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Debug info
    st.write(f"Price range: ${y_min:.2f} to ${y_max:.2f}")
else:
    st.error("No data available")
