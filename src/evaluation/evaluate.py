import numpy as np
import pandas as pd
from typing import Dict, Any

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prev_close: np.ndarray
) -> Dict[str, float]:
    """
    Compute real-life quantitative financial and machine learning metrics.
    
    Args:
        y_true: Actual closing prices on test set
        y_pred: Predicted closing prices on test set
        prev_close: Previous day's closing prices (t - 1)
    """
    # 1. Error Metrics
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)

    # 2. Directional Accuracy (Hit Ratio)
    # Does predicted direction (y_pred - prev_close) match actual direction (y_true - prev_close)?
    actual_direction = np.sign(y_true - prev_close)
    predicted_direction = np.sign(y_pred - prev_close)

    # Filter out flat days where direction is 0
    non_zero = actual_direction != 0
    if np.sum(non_zero) > 0:
        correct_directions = (predicted_direction[non_zero] == actual_direction[non_zero])
        directional_accuracy = float(np.mean(correct_directions) * 100.0)
    else:
        directional_accuracy = 50.0

    return {
        'rmse': round(rmse, 2),
        'mae': round(mae, 2),
        'mape': round(mape, 2),
        'directional_accuracy': round(directional_accuracy, 2)
    }

def backtest_trading_strategy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prev_close: np.ndarray,
    sentiment_scores: np.ndarray,
    risk_free_rate: float = 0.04
) -> Dict[str, float]:
    """
    Simulate a real-life trading strategy driven by model predictions + FinBERT sentiment.
    
    Rule:
    - If model predicts next-day return > 0 AND sentiment_score > 0: Position = Long (1.0)
    - Else: Position = Cash (0.0)
    
    Compares against Buy & Hold benchmark.
    """
    daily_returns = (y_true - prev_close) / prev_close
    predicted_returns = (y_pred - prev_close) / prev_close

    # Position signal: Long when both price model and sentiment confirm upside
    signals = ((predicted_returns > 0) & (sentiment_scores > 0)).astype(float)

    strategy_daily_returns = signals * daily_returns
    cum_strategy = np.cumprod(1 + strategy_daily_returns) - 1.0
    cum_benchmark = np.cumprod(1 + daily_returns) - 1.0

    total_strategy_return = float(cum_strategy[-1] * 100.0) if len(cum_strategy) > 0 else 0.0
    total_benchmark_return = float(cum_benchmark[-1] * 100.0) if len(cum_benchmark) > 0 else 0.0

    # Sharpe Ratio (annualized, 252 trading days)
    excess_returns = strategy_daily_returns - (risk_free_rate / 252)
    std_dev = np.std(strategy_daily_returns)
    if std_dev > 1e-6:
        sharpe_ratio = float(np.mean(excess_returns) / std_dev * np.sqrt(252))
    else:
        sharpe_ratio = 0.0

    # Max Drawdown
    wealth_index = np.cumprod(1 + strategy_daily_returns)
    previous_peaks = np.maximum.accumulate(wealth_index)
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    max_drawdown = float(np.min(drawdowns) * 100.0) if len(drawdowns) > 0 else 0.0

    # Win rate of trades taken
    trades_taken = signals > 0
    if np.sum(trades_taken) > 0:
        win_rate = float(np.mean(daily_returns[trades_taken] > 0) * 100.0)
    else:
        win_rate = 0.0

    return {
        'strategy_return_pct': round(total_strategy_return, 2),
        'benchmark_return_pct': round(total_benchmark_return, 2),
        'sharpe_ratio': round(sharpe_ratio, 2),
        'max_drawdown_pct': round(abs(max_drawdown), 2),
        'trade_win_rate_pct': round(win_rate, 2),
        'num_trades': int(np.sum(trades_taken))
    }
