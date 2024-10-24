import streamlit as st
import yfinance as yf
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

    # Verify data
    st.write("Raw Data Sample:")
    st.write(df.head())
    
    st.write("\nData Info:")
    st.write(f"Number of rows: {len(df)}")
    st.write(f"Columns: {df.columns.tolist()}")
    
    st.write("\nValue Ranges:")
    for col in ['Open', 'High', 'Low', 'Close']:
        st.write(f"{col} range: {df[col].min():.2f} to {df[col].max():.2f}")
        
    st.write("\nFirst few rows of price data:")
    st.dataframe(df[['Open', 'High', 'Low', 'Close']].head(10))

except Exception as e:
    st.error(f"Error occurred: {str(e)}")
