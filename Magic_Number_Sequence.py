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

try:
    # Download data
    end = datetime.now()
    start = end - timedelta(days=365)
    df = yf.download(ticker, start=start, end=end)

    if not df.empty:
        # Debug info first
        st.write(f"Downloaded {len(df)} rows of data")
        st.write("Data range:", df.index[0], "to", df.index[-1])
        
        # Calculate y-axis range safely
        y_min = float(df['Low'].min())
        y_max = float(df['High'].max())
        y_range = y_max - y_min
        y_padding = y_range * 0.1
        
        # Show price range
        st.write(f"Price range: ${y_min:.2f} to ${y_max:.2f}")
        
        # Create figure
        fig = go.Figure()
        
        # Add candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name=ticker
            )
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'Magic Number Sequence Analysis - {ticker}',
                x=0.5
            ),
            yaxis=dict(
                title='Price',
                side='right',
                range=[y_min - y_padding, y_max + y_padding],
                showgrid=True,
                gridcolor='LightGrey'
            ),
            xaxis=dict(
                title='Date',
                rangeslider=dict(visible=False),
                showgrid=True,
                gridcolor='LightGrey'
            ),
            height=800,
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error(f"No data available for {ticker}")
        
except Exception as e:
    st.error(f"Error occurred: {str(e)}")
    st.write("Please check if the ticker symbol is correct")
