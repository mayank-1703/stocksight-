import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from typing import List, Optional

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache")

class FinBertSentimentExtractor:
    """
    Extracts financial sentiment using ProsusAI/finbert.
    Supports Apple Silicon MPS, NVIDIA CUDA, and CPU with batched inference and disk caching.
    """
    def __init__(self, model_name: str = "ProsusAI/finbert", batch_size: int = 32):
        self.batch_size = batch_size
        
        # Hardware acceleration detection
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        print(f"[Sentiment] Loading {model_name} on device: {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_sentiment_scores(self, texts: List[str]) -> np.ndarray:
        """
        Returns an array of sentiment probabilities: shape (len(texts), 3) -> [positive, negative, neutral]
        """
        if not texts:
            return np.empty((0, 3), dtype=np.float32)

        all_probs = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)

    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'text', cache_key: Optional[str] = None) -> pd.DataFrame:
        """
        Process DataFrame with text, returning sentiment scores with caching.
        """
        if df.empty:
            df['sentiment_pos'] = 0.0
            df['sentiment_neg'] = 0.0
            df['sentiment_neu'] = 0.0
            df['sentiment_score'] = 0.0
            return df

        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}_sentiment.parquet") if cache_key else None

        if cache_file and os.path.exists(cache_file):
            try:
                cached_df = pd.read_parquet(cache_file)
                if len(cached_df) == len(df):
                    return cached_df
            except Exception:
                pass

        texts = df[text_column].astype(str).tolist()
        scores = self.get_sentiment_scores(texts)

        df = df.copy()
        # FinBERT labels: 0: positive, 1: negative, 2: neutral
        df['sentiment_pos'] = scores[:, 0]
        df['sentiment_neg'] = scores[:, 1]
        df['sentiment_neu'] = scores[:, 2]
        # Compound polarity score bounded in [-1, 1]
        df['sentiment_score'] = df['sentiment_pos'] - df['sentiment_neg']

        if cache_file:
            try:
                df.to_parquet(cache_file)
            except Exception:
                pass

        return df
