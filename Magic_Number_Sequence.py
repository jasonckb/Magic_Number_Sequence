import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go

# Set page configuration
st.set_page_config(layout="wide")

# Sidebar for input
ticker_input = st.sidebar.text_input("Enter ticker", "AAPL")

# Function to check if the ticker is formatted correctly
def check_ticker_format(ticker):
    if ticker.isdigit():
        return ticker.zfill(4) + ".HK"
    return ticker.upper()  # Added upper() to standardize ticker format

# Format the ticker input
formatted_ticker = check_ticker_format(ticker_input)
st.write("Formatted Ticker: ", formatted_ticker)

try:
    # Download 1 year of data
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=1)
    data = yf.download(formatted_ticker, start=start_date, end=end_date, progress=False)
    
    # Check if data is empty
    if data.empty:
        st.error(f"No data available for {formatted_ticker}")
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
                xaxis_rangeslider_visible=False  # Removes the range slider
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
