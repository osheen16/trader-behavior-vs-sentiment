import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
sentiment_df = pd.read_csv("fear_greed_index.csv")
trades_df = pd.read_csv("historical_data.csv")

# Preprocess datasets
# Convert 'date' in sentiment_df to datetime
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

# Convert 'Timestamp IST' in trades_df to datetime and extract date
trades_df['date'] = pd.to_datetime(trades_df['Timestamp IST'], format='%d-%m-%Y %H:%M').dt.date
trades_df['date'] = pd.to_datetime(trades_df['date'])

# Merge datasets on date (inner join for common dates)
df = pd.merge(trades_df, sentiment_df, on="date", how="inner")

print("Merged Data Preview:")
print(df.head())

# -------------------------------
# 1. Fear & Greed Index Trend
# -------------------------------
plt.figure(figsize=(12,6))
sns.lineplot(x="date", y="value", data=df, color="blue", label="Fear & Greed Index")
plt.title("Fear & Greed Index Over Time")
plt.xlabel("Date")
plt.ylabel("Index Value")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# 2. Distribution of Sentiment Classes
# -------------------------------
plt.figure(figsize=(8,6))
sns.countplot(x="classification", data=sentiment_df, palette="Set2", order=["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"])
plt.title("Distribution of Sentiment Classes")
plt.xlabel("Sentiment Classification")
plt.ylabel("Count")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# -------------------------------
# 3. Historical Trading Behavior
# -------------------------------
# Aggregate trades to daily level for price and volume
daily_trades = trades_df.groupby('date').agg({
    'Execution Price': 'mean',  # Average execution price per day
    'Size USD': 'sum'          # Total trading volume in USD per day
}).reset_index()

plt.figure(figsize=(12,6))
sns.lineplot(x="date", y="Execution Price", data=daily_trades, color="green", label="Average Execution Price")
plt.title("Average Trading Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,6))
sns.lineplot(x="date", y="Size USD", data=daily_trades, color="orange", label="Trading Volume (USD)")
plt.title("Trading Volume Over Time")
plt.xlabel("Date")
plt.ylabel("Volume (USD)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# 4. Correlation Between Sentiment & Trading
# -------------------------------
# Merge daily aggregates with sentiment data
daily_df = pd.merge(daily_trades, sentiment_df, on="date", how="inner")

# Calculate daily returns based on Execution Price
daily_df["returns"] = daily_df["Execution Price"].pct_change()

plt.figure(figsize=(8,6))
sns.scatterplot(x="value", y="returns", data=daily_df, alpha=0.5)
plt.title("Correlation between Fear & Greed Index and Daily Returns")
plt.xlabel("Fear & Greed Index")
plt.ylabel("Daily Returns")
plt.tight_layout()
plt.show()

# Correlation matrix
corr = daily_df[["value", "returns", "Size USD"]].corr()
print("Correlation Matrix:")
print(corr)

plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# -------------------------------
# 5. Trader Performance by Sentiment
# -------------------------------
# Calculate average Closed PnL by sentiment classification
pnl_by_sentiment = df.groupby('classification')['Closed PnL'].mean().reset_index()

plt.figure(figsize=(8,6))
sns.barplot(x="classification", y="Closed PnL", data=pnl_by_sentiment, palette="Set3", order=["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"])
plt.title("Average Closed PnL by Sentiment Classification")
plt.xlabel("Sentiment Classification")
plt.ylabel("Average Closed PnL (USD)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()