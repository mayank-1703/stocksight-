import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict

class DataPreprocessor:
    def __init__(self, sequence_length: int = 10):
        self.sequence_length = sequence_length
        self.feature_scaler = StandardScaler()
        self.sentiment_scaler = StandardScaler()

    def clean_text_data(self, df_news: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare text data before sending to BERT."""
        df_news = df_news.dropna(subset=['text']).copy()
        df_news['text'] = df_news['text'].astype(str).str.strip()
        return df_news

    def align_and_merge(self, df_price: pd.DataFrame, df_sentiment: pd.DataFrame) -> pd.DataFrame:
        """
        Align stock data and daily aggregated sentiment data by date.
        Ensures strict timezone neutralization to prevent merge drops.
        """
        df_price = df_price.copy()
        df_sentiment = df_sentiment.copy()

        df_price['date'] = pd.to_datetime(df_price['timestamp']).dt.tz_localize(None).dt.date
        df_sentiment['date'] = pd.to_datetime(df_sentiment['timestamp']).dt.tz_localize(None).dt.date

        # Sentiment feature columns
        sentiment_cols = ['sentiment_pos', 'sentiment_neg', 'sentiment_neu', 'sentiment_score']
        avail_sent_cols = [c for c in sentiment_cols if c in df_sentiment.columns]

        daily_sentiment = df_sentiment.groupby('date')[avail_sent_cols].mean().reset_index()

        # Drop sentiment columns from df_price if present to prevent suffix collision
        df_price_clean = df_price.drop(columns=[c for c in avail_sent_cols if c in df_price.columns], errors='ignore')

        # Merge onto price dates
        merged_df = pd.merge(df_price_clean, daily_sentiment, on='date', how='left')

        # Forward-fill weekend/holiday sentiments, then default remaining to neutral (0)
        for col in avail_sent_cols:
            merged_df[col] = merged_df[col].ffill().fillna(0.0)

        # Sort chronologically
        merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
        return merged_df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer stationary time-series features essential for financial forecasting.
        Raw prices are non-stationary I(1); returns and ratios are stationary I(0).
        """
        df = df.copy()
        close = df['Close'].values
        open_p = df['Open'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values

        # 1. Daily Return (Percentage price delta)
        returns = np.zeros(len(close))
        returns[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-6)
        df['return'] = returns

        # 2. High-Low Volatility Spread
        df['hl_spread'] = (high - low) / np.maximum(close, 1e-6)

        # 3. Close-Open Intraday Momentum
        df['co_spread'] = (close - open_p) / np.maximum(open_p, 1e-6)

        # 4. Moving Average Momentum (Close relative to 10-day SMA)
        sma_10 = df['Close'].rolling(window=10, min_periods=1).mean()
        df['sma_ratio'] = (df['Close'] - sma_10) / np.maximum(sma_10, 1e-6)

        # 5. Normalized Volume Z-Score
        vol_mean = df['Volume'].rolling(window=20, min_periods=1).mean()
        vol_std = df['Volume'].rolling(window=20, min_periods=1).std().fillna(1.0)
        df['vol_norm'] = (df['Volume'] - vol_mean) / np.maximum(vol_std, 1e-6)

        # Target: Next-day return r_{t+1}
        target_returns = np.zeros(len(close))
        target_returns[:-1] = returns[1:]
        target_returns[-1] = returns[-1]  # edge
        df['target_return'] = target_returns

        return df

    def prepare_dataset(
        self,
        merged_df: pd.DataFrame,
        train_split: float = 0.8
    ) -> Dict:
        """
        Prepare stationary, causally-aligned sequences for training and evaluation.
        
        Target at index t is the next-day return r_{t}.
        Price input is the sequence of stationary market features [t - seq_len : t].
        Sentiment input is the aggregated sentiment up to day t - 1 (zero lookahead).
        """
        df = self.engineer_features(merged_df)

        price_feature_cols = ['return', 'hl_spread', 'co_spread', 'sma_ratio', 'vol_norm']
        sentiment_feature_cols = ['sentiment_pos', 'sentiment_neg', 'sentiment_neu', 'sentiment_score']

        num_samples = len(df)
        split_idx = int(num_samples * train_split)

        # Fit scalers strictly on training portion to prevent data leakage
        train_raw = df.iloc[:split_idx]
        test_raw = df.iloc[split_idx:]

        scaled_train_price = self.feature_scaler.fit_transform(train_raw[price_feature_cols].values)
        scaled_test_price = self.feature_scaler.transform(test_raw[price_feature_cols].values)
        scaled_all_price = np.vstack([scaled_train_price, scaled_test_price])

        scaled_train_sent = self.sentiment_scaler.fit_transform(train_raw[sentiment_feature_cols].values)
        scaled_test_sent = self.sentiment_scaler.transform(test_raw[sentiment_feature_cols].values)
        scaled_all_sent = np.vstack([scaled_train_sent, scaled_test_sent])

        raw_close = df['Close'].values
        target_returns = df['target_return'].values

        # Build causal sequences
        X_price, X_sent, y_return, prev_close_list, true_next_close_list = [], [], [], [], []

        for i in range(self.sequence_length, num_samples - 1):
            # Market feature sequence: [i - seq_len : i]
            X_price.append(scaled_all_price[i - self.sequence_length:i, :])

            # Sentiment features: average over recent window [i - seq_len : i]
            window_sent = np.mean(scaled_all_sent[i - self.sequence_length:i, :], axis=0)
            X_sent.append(window_sent)

            # Target is next-day return
            y_return.append(target_returns[i - 1])

            # Ground-truth prices for dollar evaluation
            prev_close_list.append(raw_close[i - 1])
            true_next_close_list.append(raw_close[i])

        X_price = np.array(X_price, dtype=np.float32)
        X_sent = np.array(X_sent, dtype=np.float32)
        y_return = np.array(y_return, dtype=np.float32)
        prev_close_arr = np.array(prev_close_list, dtype=np.float32)
        true_next_close_arr = np.array(true_next_close_list, dtype=np.float32)

        seq_train_size = max(split_idx - self.sequence_length, 1)

        return {
            'X_price_train': X_price[:seq_train_size],
            'X_price_test': X_price[seq_train_size:],
            'X_sent_train': X_sent[:seq_train_size],
            'X_sent_test': X_sent[seq_train_size:],
            'y_train': y_return[:seq_train_size],
            'y_test': y_return[seq_train_size:],
            'prev_close_test': prev_close_arr[seq_train_size:],
            'true_next_close_test': true_next_close_arr[seq_train_size:],
            'test_dates': df['timestamp'].values[split_idx + 1:],
            'num_price_features': len(price_feature_cols),
            'num_sentiment_features': len(sentiment_feature_cols),
            'feature_scaler': self.feature_scaler,
            'sentiment_scaler': self.sentiment_scaler,
            'price_feature_cols': price_feature_cols,
            'sentiment_feature_cols': sentiment_feature_cols
        }
