import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from collections import deque

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

def download_data():
    # Calculate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=500)  # Download 500 days
    display_start = end_date - timedelta(days=365)  # Show 1 year
    
    # Convert to date objects
    display_start = display_start.date()
    end_date = end_date.date()
    start_date = start_date.date()
    
    # Download data
    df = yf.download(ticker, start=start_date, end=end_date)
    df.index = df.index.date
    
    return df, display_start, end_date

# [All helper functions remain exactly the same until create_td_sequential_chart]

def create_td_sequential_chart(df, start_date, end_date):
    buy_setup, sell_setup, buy_countdown, sell_countdown, buy_perfection, sell_perfection, buy_deferred, sell_deferred, tdst = calculate_td_sequential(df)
    
    # Clean incomplete setups
    buy_setup, sell_setup = clean_incomplete_setups(buy_setup, sell_setup)
    
    # Filter data to only include trading days within the date range
    mask = (df.index >= start_date) & (df.index <= end_date)
    df_filtered = df[mask]
    
    fig = go.Figure(data=[go.Candlestick(x=df_filtered.index,
                                        open=df_filtered['Open'],
                                        high=df_filtered['High'],
                                        low=df_filtered['Low'],
                                        close=df_filtered['Close'])])
    
    # Add Trendline levels (renamed from TDST)
    for i, (level, date) in enumerate(tdst.resistance_levels):
        fig.add_shape(
            type="line",
            x0=date,
            x1=df_filtered.index[-1],
            y0=level,
            y1=level,
            line=dict(
                color="red",
                width=2 if i == len(tdst.resistance_levels)-1 else 1,
                dash="dash"
            )
        )
    
    for i, (level, date) in enumerate(tdst.support_levels):
        fig.add_shape(
            type="line",
            x0=date,
            x1=df_filtered.index[-1],
            y0=level,
            y1=level,
            line=dict(
                color="green",
                width=2 if i == len(tdst.support_levels)-1 else 1,
                dash="dash"
            )
        )
    
    # Add annotations for filtered date range
    for i in range(len(df_filtered)):
        idx = df_filtered.index[i]
        orig_idx = df.index.get_loc(idx)
        
        # [Setup and countdown annotations remain the same]
        # [Previous annotation code remains exactly the same]
    
    # Add Trendline level labels (renamed from TDST)
    for i, (level, date) in enumerate(tdst.resistance_levels):
        fig.add_annotation(
            x=df_filtered.index[-1],
            y=level,
            text=f"TrendLine R{len(tdst.resistance_levels)-i}",
            showarrow=False,
            xanchor="left",
            xshift=10,
            font=dict(color="red", size=10)
        )
    
    for i, (level, date) in enumerate(tdst.support_levels):
        fig.add_annotation(
            x=df_filtered.index[-1],
            y=level,
            text=f"TrendLine S{len(tdst.support_levels)-i}",
            showarrow=False,
            xanchor="left",
            xshift=10,
            font=dict(color="green", size=10)
        )
    
    # Update layout
    fig.update_layout(
        title=f'Magic Number Sequence Analysis - {ticker}',
        yaxis_title='Price',
        xaxis_title='Date',
        showlegend=False,
        height=800,
        width=None,  # Let Streamlit handle the width
        yaxis=dict(
            autorange=True,
            fixedrange=False,
            range=[df_filtered['Low'].min() * 0.99, df_filtered['High'].max() * 1.01]
        ),
        xaxis=dict(
            rangeslider=dict(visible=False),
            type='category',
            tickangle=45,
            tickformat="%Y-%m-%d"
        )
    )
    
    return fig

def main():
    try:
        df, display_start, end_date = download_data()
        fig = create_td_sequential_chart(df, display_start, end_date)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

if __name__ == "__main__":
    main()
