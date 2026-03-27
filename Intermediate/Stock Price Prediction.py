"""
Stock Price Prediction
=======================
Predicts next-day stock closing price using:
- Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- LSTM neural network (via Keras/TensorFlow)
- Linear Regression baseline
- Evaluation: RMSE, MAE, MAPE
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

np.random.seed(42)

# ── Synthetic stock price (geometric Brownian motion) ─────────────────────────
def generate_stock_data(n=500, start_price=150.0, mu=0.0002, sigma=0.015):
    returns = np.random.normal(mu, sigma, n)
    prices  = start_price * np.exp(np.cumsum(returns))
    dates   = pd.date_range("2022-01-01", periods=n, freq="B")
    volume  = np.random.randint(1_000_000, 10_000_000, n)
    high    = prices * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low     = prices * (1 - np.abs(np.random.normal(0, 0.005, n)))
    return pd.DataFrame({"Date": dates, "Open": prices, "High": high,
                          "Low": low, "Close": prices, "Volume": volume}).set_index("Date")

# ── Technical indicators ──────────────────────────────────────────────────────
def add_indicators(df):
    df = df.copy()
    df["SMA_20"]  = df["Close"].rolling(20).mean()
    df["SMA_50"]  = df["Close"].rolling(50).mean()
    df["EMA_12"]  = df["Close"].ewm(span=12).mean()
    df["EMA_26"]  = df["Close"].ewm(span=26).mean()
    df["MACD"]    = df["EMA_12"] - df["EMA_26"]
    df["Signal"]  = df["MACD"].ewm(span=9).mean()

    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-10)
    df["RSI"]     = 100 - (100 / (1 + rs))

    std20          = df["Close"].rolling(20).std()
    df["BB_upper"] = df["SMA_20"] + 2 * std20
    df["BB_lower"] = df["SMA_20"] - 2 * std20
    df["BB_width"] = df["BB_upper"] - df["BB_lower"]

    df["Return_1d"] = df["Close"].pct_change()
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Volatility"]= df["Return_1d"].rolling(20).std()
    return df.dropna()

# ── Sequence builder for time-series ─────────────────────────────────────────
def make_sequences(data, look_back=30):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i])
        y.append(data[i, 0])           # first column = Close (scaled)
    return np.array(X), np.array(y)

print("=" * 60)
print("  STOCK PRICE PREDICTION")
print("=" * 60)

df = generate_stock_data(500)
df = add_indicators(df)
print(f"\nDataset rows  : {len(df)}")
print(f"Features      : {list(df.columns)}")

# ── Feature matrix ────────────────────────────────────────────────────────────
feature_cols = ["Close", "SMA_20", "SMA_50", "RSI", "MACD", "BB_width",
                "Return_1d", "Return_5d", "Volatility", "Volume"]
data = df[feature_cols].values

scaler = MinMaxScaler()
data_sc = scaler.fit_transform(data)

LOOK_BACK   = 30
TRAIN_RATIO = 0.80
split       = int(len(data_sc) * TRAIN_RATIO)

X, y = make_sequences(data_sc, LOOK_BACK)
X_train, X_test = X[:split - LOOK_BACK], X[split - LOOK_BACK:]
y_train, y_test = y[:split - LOOK_BACK], y[split - LOOK_BACK:]

print(f"\nTrain samples : {len(X_train)}")
print(f"Test samples  : {len(X_test)}")

# ── Baseline: Linear Regression on last-day features ─────────────────────────
lr = LinearRegression()
lr.fit(X_train[:, -1, :], y_train)
lr_preds = lr.predict(X_test[:, -1, :])

def inverse_close(scaled_vals):
    dummy = np.zeros((len(scaled_vals), data.shape[1]))
    dummy[:, 0] = scaled_vals
    return scaler.inverse_transform(dummy)[:, 0]

lr_preds_real = inverse_close(lr_preds)
y_test_real   = inverse_close(y_test)

lr_rmse = np.sqrt(mean_squared_error(y_test_real, lr_preds_real))
lr_mae  = mean_absolute_error(y_test_real, lr_preds_real)
lr_mape = np.mean(np.abs((y_test_real - lr_preds_real) / y_test_real)) * 100

print(f"\n── Linear Regression (baseline) ──")
print(f"  RMSE : {lr_rmse:.4f}")
print(f"  MAE  : {lr_mae:.4f}")
print(f"  MAPE : {lr_mape:.2f}%")

# ── LSTM (if TensorFlow available) ───────────────────────────────────────────
lstm_available = False
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    lstm_available = True
except ImportError:
    print("\n[INFO] TensorFlow not installed — skipping LSTM. Run: pip install tensorflow")

if lstm_available:
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOK_BACK, len(feature_cols))),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.summary()

    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=50, batch_size=32,
        validation_split=0.1,
        callbacks=[es], verbose=1,
    )

    lstm_preds_sc = model.predict(X_test).flatten()
    lstm_preds    = inverse_close(lstm_preds_sc)

    lstm_rmse = np.sqrt(mean_squared_error(y_test_real, lstm_preds))
    lstm_mae  = mean_absolute_error(y_test_real, lstm_preds)
    lstm_mape = np.mean(np.abs((y_test_real - lstm_preds) / y_test_real)) * 100

    print(f"\n── LSTM ──")
    print(f"  RMSE : {lstm_rmse:.4f}")
    print(f"  MAE  : {lstm_mae:.4f}")
    print(f"  MAPE : {lstm_mape:.2f}%")

# ── Visualisations ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Stock Price Prediction Dashboard", fontsize=15, fontweight="bold")

# Price + moving averages
axes[0, 0].plot(df.index[-200:], df["Close"].values[-200:], label="Close", linewidth=1)
axes[0, 0].plot(df.index[-200:], df["SMA_20"].values[-200:], label="SMA 20", linestyle="--")
axes[0, 0].plot(df.index[-200:], df["SMA_50"].values[-200:], label="SMA 50", linestyle="--")
axes[0, 0].set_title("Price & Moving Averages")
axes[0, 0].legend(); axes[0, 0].set_ylabel("Price ($)")

# RSI
axes[0, 1].plot(df.index[-200:], df["RSI"].values[-200:], color="#e74c3c")
axes[0, 1].axhline(70, linestyle="--", color="gray"); axes[0, 1].axhline(30, linestyle="--", color="gray")
axes[0, 1].set_title("RSI (14)"); axes[0, 1].set_ylabel("RSI")

# LR predictions
axes[1, 0].plot(y_test_real, label="Actual", linewidth=1.5)
axes[1, 0].plot(lr_preds_real, label="LR Predicted", linewidth=1.5, linestyle="--")
axes[1, 0].set_title(f"Linear Regression  |  RMSE={lr_rmse:.2f}  MAPE={lr_mape:.2f}%")
axes[1, 0].legend()

# LSTM predictions (if available)
if lstm_available:
    axes[1, 1].plot(y_test_real, label="Actual", linewidth=1.5)
    axes[1, 1].plot(lstm_preds, label="LSTM Predicted", linewidth=1.5, linestyle="--", color="green")
    axes[1, 1].set_title(f"LSTM  |  RMSE={lstm_rmse:.2f}  MAPE={lstm_mape:.2f}%")
    axes[1, 1].legend()
else:
    axes[1, 1].text(0.5, 0.5, "LSTM requires TensorFlow\npip install tensorflow",
                    ha="center", va="center", fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].set_title("LSTM (not available)")

plt.tight_layout()
plt.savefig("stock_prediction_results.png", dpi=150, bbox_inches="tight")
print("\nPlots saved → stock_prediction_results.png")
print("\nStock price prediction complete!")