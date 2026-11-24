from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from src.data.ingestion import fetch_stock_data, fetch_financial_news
from src.data.preprocessing import DataPreprocessor
from src.models.sentiment import FinBertSentimentExtractor
from src.models.hybrid_model import HybridStockPredictor

router = APIRouter()
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

class PredictionRequest(BaseModel):
    ticker: str
    custom_news: Optional[str] = None

class PredictionResponse(BaseModel):
    ticker: str
    current_price: float
    predicted_next_day_price: float
    predicted_change_pct: float
    predicted_direction: str
    sentiment_analysis: Dict[str, Any]
    status: str

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "StockSight++",
        "version": "2.0.0",
        "device": "mps" if os.uname().sysname == "Darwin" else "cpu"
    }

@router.get("/metrics/{ticker}")
async def get_model_metrics(ticker: str):
    """Retrieve the real evaluated benchmark metrics for a trained ticker."""
    metrics_path = os.path.join(MODELS_DIR, f"{ticker.upper()}_metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(
            status_code=404,
            detail=f"No benchmark metrics found for {ticker.upper()}. Run training first: python -m src.training.train --ticker {ticker.upper()}"
        )
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    return data

@router.post("/predict", response_model=PredictionResponse)
async def predict_stock(request: PredictionRequest):
    """
    Live real-time inference workflow:
    1. Ingest latest market session price series
    2. Extract FinBERT sentiment from real/custom news
    3. Pass through trained multimodal hybrid LSTM+BERT model
    4. Return predicted price, directional signal, and sentiment confidence
    """
    ticker = request.ticker.upper()
    model_path = os.path.join(MODELS_DIR, f"{ticker}_hybrid_model.pt")

    # Ingest recent price data
    try:
        df_price = fetch_stock_data(ticker, period="6mo")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch market data: {str(e)}")

    current_price = float(df_price['Close'].iloc[-1])
    seq_len = 10

    # Ingest / Score Sentiment
    if request.custom_news:
        news_df = pd.DataFrame([{
            'timestamp': datetime.now(),
            'ticker': ticker,
            'text': request.custom_news,
            'source': 'user_input'
        }])
    else:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=15)
        news_df = fetch_financial_news(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    sentiment_extractor = FinBertSentimentExtractor(batch_size=16)
    news_df_scored = sentiment_extractor.process_dataframe(news_df)

    latest_pos = float(news_df_scored['sentiment_pos'].mean())
    latest_neg = float(news_df_scored['sentiment_neg'].mean())
    latest_neu = float(news_df_scored['sentiment_neu'].mean())
    sentiment_score = latest_pos - latest_neg

    sentiment_label = "Bullish" if sentiment_score > 0.15 else ("Bearish" if sentiment_score < -0.15 else "Neutral")

    # If model exists, perform neural inference
    if os.path.exists(model_path):
        preprocessor = DataPreprocessor(sequence_length=seq_len)
        merged_df = preprocessor.align_and_merge(df_price, news_df_scored)
        dataset = preprocessor.prepare_dataset(merged_df, train_split=0.8)

        model = HybridStockPredictor(
            sequence_length=seq_len,
            num_price_features=dataset['num_price_features'],
            num_sentiment_features=dataset['num_sentiment_features']
        )
        model.load(model_path)

        # Build inference sequence from most recent window
        feat_df = preprocessor.engineer_features(merged_df)
        price_feats = dataset['price_feature_cols']
        sent_feats = dataset['sentiment_feature_cols']

        last_price_window = feat_df[price_feats].iloc[-seq_len:].values
        scaled_price_window = dataset['feature_scaler'].transform(last_price_window)
        scaled_price_window = np.expand_dims(scaled_price_window, axis=0) # shape (1, seq_len, 5)

        last_sent_window = feat_df[sent_feats].iloc[-seq_len:].values
        scaled_sent_window = dataset['sentiment_scaler'].transform(last_sent_window)
        scaled_sent_mean = np.mean(scaled_sent_window, axis=0, keepdims=True) # shape (1, 4)

        pred_return = float(model.predict(scaled_price_window, scaled_sent_mean)[0])
        # Bound extreme single-day return projections to realistic market range (+-5%)
        pred_return = max(min(pred_return, 0.05), -0.05)
        raw_pred = current_price * (1.0 + pred_return)
        status_msg = "Inference completed via trained Hybrid LSTM+FinBERT model."
    else:
        # Fallback heuristic based on recent momentum + sentiment signal
        daily_return = (df_price['Close'].iloc[-1] - df_price['Close'].iloc[-5]) / df_price['Close'].iloc[-5] / 5.0
        sentiment_delta = sentiment_score * 0.006
        pred_return = daily_return * 0.4 + sentiment_delta
        raw_pred = current_price * (1.0 + pred_return)
        status_msg = f"Model {model_path} not found. Returning momentum+sentiment projection. Run train.py to train checkpoint."

    predicted_change_pct = round(((raw_pred - current_price) / current_price) * 100.0, 2)
    predicted_direction = "UP" if raw_pred >= current_price else "DOWN"

    return PredictionResponse(
        ticker=ticker,
        current_price=round(current_price, 2),
        predicted_next_day_price=round(float(raw_pred), 2),
        predicted_change_pct=predicted_change_pct,
        predicted_direction=predicted_direction,
        sentiment_analysis={
            "signal": sentiment_label,
            "compound_score": round(sentiment_score, 4),
            "positive_prob": round(latest_pos, 4),
            "negative_prob": round(latest_neg, 4),
            "neutral_prob": round(latest_neu, 4)
        },
        status=status_msg
    )
