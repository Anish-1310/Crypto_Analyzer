import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
from datetime import datetime

# 1. Fetch Real-time and Historical Data
def fetch_crypto_data(crypto_id="bitcoin", days="100"):
    """
    Fetches historical market data from CoinGecko API.
    """
    print(f"Fetching data for {crypto_id} from CoinGecko...")
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart?vs_currency=usd&days={days}"
    
    # Adding a User-Agent header is good practice when making API requests
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Raise an exception for bad status codes
        data = response.json()
        
        # Extract prices (data format: [[timestamp, price], ...])
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        
        # Convert timestamp to human-readable date
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('date', inplace=True)
        df.drop(columns=['timestamp'], inplace=True)
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching data: {e}")
        print("Please check your internet connection or try again later.")
        return None

# 2. Perform Data Analysis (Moving Averages & Volatility)
def analyze_data(df):
    """
    Calculates moving averages and volatility for the dataset.
    """
    print("Analyzing data (Moving Averages & Volatility)...")
    # Clean data: Resample to daily frequency (mean price per day)
    df = df.resample('D').mean()
    
    # Calculate Moving Averages (7-day and 30-day)
    df['MA7'] = df['price'].rolling(window=7).mean()
    df['MA30'] = df['price'].rolling(window=30).mean()
    
    # Calculate Volatility (Standard Deviation of daily returns)
    df['daily_return'] = df['price'].pct_change()
    volatility = df['daily_return'].std() * np.sqrt(365) # Annualized volatility
    
    return df, volatility

# 3. Predict Short-Term Price Trend (Machine Learning)
def predict_trend(df):
    """
    Uses Linear Regression to predict the next day's price trend.
    """
    print("Training Machine Learning model (Linear Regression)...")
    
    # Prepare data for ML model
    # We will use past prices to predict future prices
    data = df.dropna().copy()
    
    # Create Lag Features (e.g., previous days' prices)
    data['price_lag_1'] = data['price'].shift(1)
    data['price_lag_2'] = data['price'].shift(2)
    data['MA7_lag_1'] = data['MA7'].shift(1)
    
    # Drop rows with NaN values created by shifting
    data.dropna(inplace=True)
    
    X = data[['price_lag_1', 'price_lag_2', 'MA7_lag_1']]
    y = data['price']
    
    # Initialize and train Linear Regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict tomorrow's price using the latest available data
    latest_data = pd.DataFrame({
        'price_lag_1': [data['price'].iloc[-1]],
        'price_lag_2': [data['price'].iloc[-2]],
        'MA7_lag_1': [data['MA7'].iloc[-1]]
    })
    
    predicted_price = model.predict(latest_data)[0]
    current_price = data['price'].iloc[-1]
    
    # Determine predicted trend direction
    direction = "Up ⬆️ (Bullish)" if predicted_price > current_price else "Down ⬇️ (Bearish)"
    
    return predicted_price, current_price, direction

# 4. Sentiment Analysis
def analyze_sentiment(headlines):
    """
    Analyzes sentiment of provided text using TextBlob Natural Language Processing.
    """
    print("Running NLP Sentiment Analysis on news headlines...")
    sentiments = []
    
    for headline in headlines:
        analysis = TextBlob(headline)
        # Polarity score is between -1 (Very Negative) and 1 (Very Positive)
        score = analysis.sentiment.polarity
        
        if score > 0.1:
            category = "Positive"
        elif score < -0.1:
            category = "Negative"
        else:
            category = "Neutral"
            
        sentiments.append((headline, category, score))
        
    # Calculate overall sentiment average
    avg_score = sum([s[2] for s in sentiments]) / len(sentiments)
    
    if avg_score > 0.1:
        overall = "Positive 🟢"
    elif avg_score < -0.1:
        overall = "Negative 🔴"
    else:
        overall = "Neutral ⚪"
    
    return sentiments, overall

# 5. Visualize the Data
def plot_data(df, crypto_name="Bitcoin"):
    """
    Plots the price and moving averages using Matplotlib.
    """
    print("Generating Matplotlib visualization...")
    plt.figure(figsize=(12, 6))
    
    # Plot real daily price
    plt.plot(df.index, df['price'], label='Daily Price', color='blue', alpha=0.6, linewidth=2)
    # Plot 7-day Moving Average
    plt.plot(df.index, df['MA7'], label='7-Day MA', color='orange', linestyle='--', linewidth=2)
    # Plot 30-day Moving Average
    plt.plot(df.index, df['MA30'], label='30-Day MA', color='red', linestyle='--', linewidth=2)
    
    plt.title(f"{crypto_name} Price Trend & Moving Averages (Past 100 Days)", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price (USD)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot instead of displaying it interactively
    filename = f"{crypto_name.lower()}_analysis_chart.png"
    plt.savefig(filename, bbox_inches='tight')
    print(f"✅ Plot successfully saved as '{filename}'")
    plt.close()

# Main Execution Flow
def main():
    print("=" * 50)
    print("🚀 Beginner-Friendly Crypto Analyzer AI 🚀")
    print("=" * 50)
    
    crypto = "bitcoin"  # You can change this to 'ethereum', 'dogecoin', etc.
    
    # Step 1: Fetch
    df = fetch_crypto_data(crypto_id=crypto, days="100")
    if df is None:
        return
        
    # Step 2: Analyze
    df, volatility = analyze_data(df)
    
    # Step 3: Predict using Machine Learning
    predicted_price, current_price, expected_direction = predict_trend(df)
    
    # Step 4: NLP Sentiment Analysis
    # We use sample headlines to keep it simple and avoid needing a paid API key for news APIs
    sample_headlines = [
        "Bitcoin surges past major resistance level, investors are thrilled!",
        "Regulatory fears cause panic selling in the crypto market.",
        "Ethereum network upgrade scheduled for next month is proceeding as planned.",
        "Whales are accumulating more Bitcoin amid market uncertainty.",
        "Major exchange hacked, millions lost but the market remains stable."
    ]
    _, overall_sentiment = analyze_sentiment(sample_headlines)
    
    # Step 5: Visualize the data to a PNG image
    plot_data(df, crypto_name=crypto.capitalize())
    
    # Step 6: Print Summary Report
    print("\n" + "=" * 50)
    print("🎯 FINAL AI INSIGHTS & SUMMARY 🎯")
    print("=" * 50)
    print(f"💰 Asset Analysed:  {crypto.capitalize()}")
    print(f"📊 Current Price:   ${current_price:,.2f}")
    print(f"📈 7-Day MA:        ${df['MA7'].iloc[-1]:,.2f}")
    print(f"🌊 Volatility:      {volatility:.2%} (Annualized)")
    print("-" * 50)
    print("🤖 MACHINE LEARNING PREDICTION (Linear Regression)")
    print(f"🔮 Predicted Price: ${predicted_price:,.2f}")
    print(f"📉 Trend Direction: {expected_direction}")
    print("-" * 50)
    print("📰 NATURAL LANGUAGE PROCESSING SENTIMENT")
    print(f"🗣️ Market Mood:     {overall_sentiment}")
    print("=" * 50)

if __name__ == "__main__":
    main()
