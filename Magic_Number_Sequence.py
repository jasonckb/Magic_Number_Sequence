import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="Magic Number Sequence by Jason Chan",
    layout="wide"
)

# Title
st.title("Magic Number Sequence by Jason Chan")

# Sidebar for ticker input
with st.sidebar:
    ticker = st.text_input("Enter Stock Symbol", "AAPL")
    
    # Process HK stocks
    if ticker.isdigit():
        ticker = f"{int(ticker):04d}.HK"

def create_basic_chart(df):
    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=ticker
        )
    ])
    
    fig.update_layout(
        title=f'Magic Number Sequence Analysis - {ticker}',
        yaxis_title='Price',
        xaxis_title='Date',
        height=800,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig

def main():
    try:
        # Download one year of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Download data
        st.write("Downloading data...")
        df = yf.download(
            ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d')
        )
        
        if not df.empty:
            st.write("Data downloaded successfully")
            st.write("Number of rows:", len(df))
            st.write("Date range:", df.index[0], "to", df.index[-1])
            
            # Create and show chart
            fig = create_basic_chart(df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No data available for the selected stock")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
