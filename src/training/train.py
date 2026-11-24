import os
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn

from src.data.ingestion import fetch_stock_data, fetch_financial_news
from src.data.preprocessing import DataPreprocessor
from src.models.sentiment import FinBertSentimentExtractor
from src.models.hybrid_model import HybridStockPredictor, TorchPriceOnlyModel
from src.evaluation.evaluate import evaluate_predictions, backtest_trading_strategy

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

def train_and_evaluate(
    ticker: str = "AAPL",
    period: str = "2y",
    epochs: int = 35,
    seq_len: int = 10,
    batch_size: int = 16
):
    print("=" * 70)
    print(f"🚀 StockSight++ Quantitative Pipeline: Training & Real-Life Benchmark for {ticker.upper()}")
    print("=" * 70)

    # 1. Fetch Market Data
    print(f"\n[1/5] Ingesting historical market data for {ticker.upper()} (period: {period})...")
    df_price = fetch_stock_data(ticker, period=period)
    print(f"  ✓ Fetched {len(df_price)} trading sessions ({df_price['timestamp'].min().date()} to {df_price['timestamp'].max().date()})")

    # 2. Ingest Financial News & Compute FinBERT Sentiment
    print(f"\n[2/5] Ingesting financial news and extracting FinBERT sentiment embeddings...")
    start_date = df_price['timestamp'].min().strftime("%Y-%m-%d")
    end_date = df_price['timestamp'].max().strftime("%Y-%m-%d")
    df_news = fetch_financial_news(ticker, start_date=start_date, end_date=end_date)
    print(f"  ✓ Processed {len(df_news)} financial headlines")

    sentiment_extractor = FinBertSentimentExtractor(batch_size=32)
    df_news_sentiment = sentiment_extractor.process_dataframe(
        df_news,
        cache_key=f"{ticker}_{start_date}_{end_date}"
    )
    print(f"  ✓ Extracted FinBERT probabilities [Positive, Negative, Neutral, Polarity]")

    # 3. Preprocessing & Stationary Sequence Generation
    print(f"\n[3/5] Engineering stationary features & causal sequences (seq_len={seq_len})...")
    preprocessor = DataPreprocessor(sequence_length=seq_len)
    merged_df = preprocessor.align_and_merge(df_price, df_news_sentiment)

    dataset = preprocessor.prepare_dataset(merged_df, train_split=0.8)

    X_p_train, X_p_test = dataset['X_price_train'], dataset['X_price_test']
    X_s_train, X_s_test = dataset['X_sent_train'], dataset['X_sent_test']
    y_train, y_test = dataset['y_train'], dataset['y_test']
    prev_close_test = dataset['prev_close_test']
    true_next_close_test = dataset['true_next_close_test']

    num_price_feats = dataset['num_price_features']
    num_sent_feats = dataset['num_sentiment_features']

    print(f"  ✓ Training sequences: {len(X_p_train)} | Out-of-sample Test sequences: {len(X_p_test)}")

    # 4. Train Baseline (Price-Only LSTM) to Measure Exact Sentiment Lift
    print(f"\n[4/5] Training Baseline (Price-Only LSTM) vs. Multimodal Hybrid (Price + FinBERT)...")
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Train Baseline Model
    print("  --> Training Baseline Model (Price-Only LSTM)...")
    base_model = TorchPriceOnlyModel(sequence_length=seq_len, num_price_features=num_price_feats).to(device)
    base_opt = torch.optim.Adam(base_model.parameters(), lr=0.001)
    base_crit = nn.MSELoss()

    t_p_train = torch.tensor(X_p_train, dtype=torch.float32)
    t_y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    
    base_model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(X_p_train))
        for i in range(0, len(X_p_train), batch_size):
            idxs = perm[i:i + batch_size]
            bx, by = t_p_train[idxs].to(device), t_y_train[idxs].to(device)
            base_opt.zero_grad()
            loss = base_crit(base_model(bx), by)
            loss.backward()
            base_opt.step()

    base_model.eval()
    with torch.no_grad():
        base_pred_returns = base_model(torch.tensor(X_p_test, dtype=torch.float32).to(device)).cpu().numpy().flatten()
    base_pred_prices = prev_close_test * (1.0 + base_pred_returns)
    base_metrics = evaluate_predictions(true_next_close_test, base_pred_prices, prev_close_test)

    # Train Hybrid Model (Price LSTM + FinBERT Dense)
    print("  --> Training Multimodal Hybrid Model (Price LSTM + FinBERT Dense)...")
    hybrid = HybridStockPredictor(
        sequence_length=seq_len,
        num_price_features=num_price_feats,
        num_sentiment_features=num_sent_feats,
        lr=0.001
    )
    hybrid.fit(X_p_train, X_s_train, y_train, epochs=epochs, batch_size=batch_size, verbose=False)

    hybrid_pred_returns = hybrid.predict(X_p_test, X_s_test)
    hybrid_pred_prices = prev_close_test * (1.0 + hybrid_pred_returns)
    hybrid_metrics = evaluate_predictions(true_next_close_test, hybrid_pred_prices, prev_close_test)

    # 5. Sentiment Lift & Backtest Simulation
    print(f"\n[5/5] Backtesting Trading Strategy & Quantifying Real-Life Metrics...")
    test_sentiment_scores = X_s_test[:, 3] # sentiment_score column
    backtest_results = backtest_trading_strategy(
        true_next_close_test,
        hybrid_pred_prices,
        prev_close_test,
        test_sentiment_scores
    )

    # Compute Sentiment Lift
    rmse_lift = round(((base_metrics['rmse'] - hybrid_metrics['rmse']) / base_metrics['rmse']) * 100.0, 2)
    da_lift = round(hybrid_metrics['directional_accuracy'] - base_metrics['directional_accuracy'], 2)

    # Save artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_save_path = os.path.join(MODELS_DIR, f"{ticker.upper()}_hybrid_model.pt")
    hybrid.save(model_save_path)

    report = {
        'ticker': ticker.upper(),
        'period': period,
        'evaluation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'test_sessions': len(true_next_close_test),
        'metrics': {
            'hybrid': hybrid_metrics,
            'baseline_price_only': base_metrics,
            'sentiment_lift': {
                'rmse_reduction_pct': rmse_lift,
                'directional_accuracy_gain_pct': da_lift
            },
            'trading_strategy': backtest_results
        }
    }

    metrics_file = os.path.join(MODELS_DIR, f"{ticker.upper()}_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(report, f, indent=4)

    # Print Formatted Benchmark Table
    print("\n" + "=" * 70)
    print(f"📊 REAL-LIFE PERFORMANCE BENCHMARK REPORT: {ticker.upper()}")
    print("=" * 70)
    print(f"{'Quantitative Metric':<32} | {'Price-Only LSTM':<16} | {'StockSight++ (Hybrid)':<16}")
    print("-" * 70)
    print(f"{'Root Mean Squared Error (RMSE)':<32} | ${base_metrics['rmse']:<15.2f} | ${hybrid_metrics['rmse']:<15.2f}")
    print(f"{'Mean Absolute Error (MAE)':<32} | ${base_metrics['mae']:<15.2f} | ${hybrid_metrics['mae']:<15.2f}")
    print(f"{'Mean Abs % Error (MAPE)':<32} | {base_metrics['mape']:<15.2f}% | {hybrid_metrics['mape']:<15.2f}%")
    print(f"{'Directional Accuracy (Hit Rate)':<32} | {base_metrics['directional_accuracy']:<15.2f}% | {hybrid_metrics['directional_accuracy']:<15.2f}%")
    print("-" * 70)
    print(f"🎯 FinBERT Sentiment Lift: RMSE error reduced by {rmse_lift}% | Hit Rate gain: +{da_lift}%")
    print("-" * 70)
    print(f"💰 Simulated Strategy Return: {backtest_results['strategy_return_pct']}% (vs. Buy & Hold Benchmark: {backtest_results['benchmark_return_pct']}%)")
    print(f"📈 Annualized Sharpe Ratio: {backtest_results['sharpe_ratio']} | Max Drawdown: {backtest_results['max_drawdown_pct']}%")
    print(f"🏆 Trade Win Rate: {backtest_results['trade_win_rate_pct']}% across {backtest_results['num_trades']} trading signals")
    print("=" * 70)
    print(f"Artifacts Saved:")
    print(f"  • Model: {model_save_path}")
    print(f"  • Report: {metrics_file}\n")

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StockSight++ Real-Life Pipeline & Benchmark")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol (e.g., AAPL, TSLA, NVDA, MSFT)")
    parser.add_argument("--period", type=str, default="2y", help="Historical data window (e.g. 1y, 2y, 5y)")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--seq_len", type=int, default=10, help="Temporal sequence length (days)")
    args = parser.parse_args()

    train_and_evaluate(
        ticker=args.ticker,
        period=args.period,
        epochs=args.epochs,
        seq_len=args.seq_len
    )
