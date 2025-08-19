# preprocess.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess():
    # Load datasets
    sentiment_df = pd.read_csv("fear_greed_index.csv")
    trades_df = pd.read_csv("historical_data.csv")

    # Show first few rows
    print("Sentiment Data:")
    print(sentiment_df.head())
    print("\nTrade Data:")
    print(trades_df.head())

    # --- Preprocessing ---
    # Convert 'date' column in sentiment data
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], errors='coerce')

    # Convert 'Timestamp' in trade data → date
    trades_df['date'] = pd.to_datetime(trades_df['Timestamp'], unit='s', errors='coerce')

    # Sort by date
    sentiment_df = sentiment_df.sort_values('date')
    trades_df = trades_df.sort_values('date')

    # Merge on date
    merged_df = pd.merge(trades_df, sentiment_df, on="date", how="inner")

    # Save merged data
    merged_df.to_csv("processed_data.csv", index=False)
    print("\n✅ Preprocessing complete. Saved as processed_data.csv")

    return merged_df


if __name__ == "__main__":
    merged_data = load_and_preprocess()

    # Simple visualization example
    plt.figure(figsize=(10, 5))
    sns.lineplot(x="date", y="Execution Price", data=merged_data, label="Trade Execution Price")
    sns.lineplot(x="date", y="value", data=merged_data, label="Fear & Greed Index")
    plt.legend()
    plt.title("Execution Price vs Sentiment Index")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
