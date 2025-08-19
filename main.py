import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
sentiment_df = pd.read_csv("fear_greed_index.csv")
trades_df = pd.read_csv("historical_data.csv")

print(sentiment_df.head())
print(trades_df.head())
