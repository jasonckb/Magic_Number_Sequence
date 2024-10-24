import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go

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
    
    # Clean up column names - remove ticker suffix
    df.columns = [col.replace(f' {ticker}', '') for col in df.columns]
    
    # Verify data structure
    st.write("Raw Data Sample:")
    st.write(df.head())
    
    st.write("\nData Info:")
    st.write(f"Number of rows: {len(df)}")
    st.write(f"Columns: {df.columns.tolist()}")
    
    if not df.empty:
        # Create figure
        fig = go.Figure(data=[
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close']
            )
        ])
        
        # Calculate price range for y-axis
        y_min = df['Low'].min()
        y_max = df['High'].max()
        y_padding = (y_max - y_min) * 0.1
        
        # Update layout
        fig.update_layout(
            title=f'Magic Number Sequence Analysis - {ticker}',
            yaxis=dict(
                title='Price',
                range=[y_min - y_padding, y_max + y_padding],
                side='right'
            ),
            xaxis=dict(
                title='Date',
                rangeslider=dict(visible=False)
            ),
            height=800,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
    
except Exception as e:
    st.error(f"Error occurred: {str(e)}")
    st.write("Full error:", str(e))
