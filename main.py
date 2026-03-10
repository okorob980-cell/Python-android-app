import pandas as pd
import numpy as np
import yfinance as yf
import time

# Settings
SYMBOL = "BTC-USD"
FAST_MA = 20
SLOW_MA = 50
RSI_PERIOD = 14


def get_data():
    """Download market data with error handling."""
    try:
        data = yf.download(SYMBOL, period="1d", interval="1m", auto_adjust=True, progress=False)

        if data is None or data.empty:
            raise ValueError("No data returned from yfinance.")

        # Flatten MultiIndex columns (yfinance v0.2+)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]

        # Ensure standard column names exist
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in data.columns:
                raise ValueError("Missing column: " + col)

        # Convert all columns to float to avoid dtype issues
        for col in required:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        data.dropna(subset=['Close'], inplace=True)

        if data.empty:
            raise ValueError("Data is empty after cleaning.")

        return data

    except Exception as e:
        print("[ERROR] Failed to fetch data: " + str(e))
        return None


def calculate_rsi(series, period=RSI_PERIOD):
    """Calculate RSI from a plain Series."""
    # Make sure it's a flat Series
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    series = series.astype(float)
    delta = series.diff()

    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()

    # Avoid division by zero
    loss_safe = loss.replace(0, np.nan)
    rs = gain / loss_safe
    rsi = 100 - (100 / (1 + rs))

    return rsi


def generate_signals(data):
    """Generate BUY, SELL, or HOLD signal."""
    if len(data) < SLOW_MA:
        print("[WARNING] Not enough data points (" + str(len(data)) + ") for SLOW_MA (" + str(SLOW_MA) + "). Returning HOLD.")
        return "HOLD"

    # Work on a clean copy
    df = data.copy()

    # Extract Close as a plain float Series
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.astype(float)

    df['fast_ma'] = close.rolling(FAST_MA).mean()
    df['slow_ma'] = close.rolling(SLOW_MA).mean()
    df['RSI'] = calculate_rsi(close)

    latest = df.iloc[-1]

    fast = latest['fast_ma']
    slow = latest['slow_ma']
    rsi  = latest['RSI']

    # Guard against NaN values
    try:
        fast_nan = pd.isna(fast)
        slow_nan = pd.isna(slow)
        rsi_nan  = pd.isna(rsi)
    except Exception:
        print("[WARNING] Could not evaluate NaN check. Returning HOLD.")
        return "HOLD"

    if fast_nan or slow_nan or rsi_nan:
        print("[WARNING] NaN values detected in indicators. Returning HOLD.")
        return "HOLD"

    fast = float(fast)
    slow = float(slow)
    rsi  = float(rsi)

    if fast > slow and rsi < 70:
        return "BUY"
    elif fast < slow and rsi > 30:
        return "SELL"
    else:
        return "HOLD"


def trading_bot():
    """Main trading loop."""
    print("[INFO] Starting trading bot for " + SYMBOL)
    print("[INFO] Fast MA: " + str(FAST_MA) + " | Slow MA: " + str(SLOW_MA) + " | RSI Period: " + str(RSI_PERIOD))

    while True:
        try:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            print("\n[" + timestamp + "] Fetching data...")

            data = get_data()

            if data is None:
                print("[WARNING] Skipping this cycle due to data fetch failure.")
                time.sleep(60)
                continue

            signal = generate_signals(data)
            print("Signal: " + signal)

            if signal == "BUY":
                print("[ACTION] Buying asset")
            elif signal == "SELL":
                print("[ACTION] Selling asset")
            else:
                print("[ACTION] Holding — no trade executed.")

        except KeyboardInterrupt:
            print("\n[INFO] Bot stopped by user.")
            break
        except Exception as e:
            print("[ERROR] Unexpected error in trading loop: " + str(e))

        time.sleep(60)


if __name__ == "__main__":
    trading_bot()
