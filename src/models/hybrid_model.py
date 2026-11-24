import os
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
import numpy as np

class TorchHybridModel(nn.Module):
    """
    Deep Neural Network combining:
    1. Multi-layer LSTM for price time-series sequences
    2. Deep Dense layers for FinBERT sentiment embeddings
    3. Multimodal fusion head for short-term price prediction
    """
    def __init__(
        self,
        sequence_length: int,
        num_price_features: int,
        num_sentiment_features: int,
        hidden_dim: int = 64,
        num_lstm_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_price_features = num_price_features
        self.num_sentiment_features = num_sentiment_features

        # 1. Price Time-Series Branch (LSTM)
        self.lstm = nn.LSTM(
            input_size=num_price_features,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0
        )
        self.price_dropout = nn.Dropout(dropout)

        # 2. Sentiment Branch (Dense)
        self.sentiment_branch = nn.Sequential(
            nn.Linear(num_sentiment_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # 3. Multimodal Fusion Head
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 16, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x_price: torch.Tensor, x_sentiment: torch.Tensor) -> torch.Tensor:
        # x_price shape: (batch_size, seq_len, num_price_features)
        lstm_out, _ = self.lstm(x_price)
        # Take the final temporal state
        price_rep = self.price_dropout(lstm_out[:, -1, :])

        # x_sentiment shape: (batch_size, num_sentiment_features)
        sent_rep = self.sentiment_branch(x_sentiment)

        # Concatenate embeddings
        fused = torch.cat([price_rep, sent_rep], dim=-1)
        return self.fusion(fused)


class TorchPriceOnlyModel(nn.Module):
    """Baseline Model: LSTM processing ONLY historical prices (no sentiment)."""
    def __init__(self, sequence_length: int, num_price_features: int, hidden_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_price_features,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_price: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x_price)
        return self.fc(lstm_out[:, -1, :])


class HybridStockPredictor:
    """
    Production-ready wrapper for training, predicting, and persisting the hybrid model.
    """
    def __init__(
        self,
        sequence_length: int = 10,
        num_price_features: int = 5,
        num_sentiment_features: int = 4,
        lr: float = 0.001
    ):
        self.sequence_length = sequence_length
        self.num_price_features = num_price_features
        self.num_sentiment_features = num_sentiment_features
        
        # Select device (Apple Silicon MPS, CUDA, or CPU)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = TorchHybridModel(
            sequence_length=sequence_length,
            num_price_features=num_price_features,
            num_sentiment_features=num_sentiment_features
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def fit(
        self,
        X_price: np.ndarray,
        X_sent: np.ndarray,
        y: np.ndarray,
        epochs: int = 25,
        batch_size: int = 16,
        verbose: bool = True
    ) -> Dict[str, list]:
        """Train the model with PyTorch."""
        self.model.train()
        num_samples = len(X_price)
        history = {'loss': []}

        t_price = torch.tensor(X_price, dtype=torch.float32)
        t_sent = torch.tensor(X_sent, dtype=torch.float32)
        t_y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

        for epoch in range(1, epochs + 1):
            permutation = torch.randperm(num_samples)
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, num_samples, batch_size):
                indices = permutation[i:i + batch_size]
                b_price = t_price[indices].to(self.device)
                b_sent = t_sent[indices].to(self.device)
                b_y = t_y[indices].to(self.device)

                self.optimizer.zero_grad()
                preds = self.model(b_price, b_sent)
                loss = self.criterion(preds, b_y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            history['loss'].append(avg_loss)
            if verbose and (epoch % 5 == 0 or epoch == epochs):
                print(f"Epoch {epoch:02d}/{epochs:02d} | Train MSE Loss: {avg_loss:.6f}")

        return history

    def predict(self, X_price: np.ndarray, X_sent: np.ndarray) -> np.ndarray:
        """Generate predictions for price and sentiment sequences."""
        self.model.eval()
        with torch.no_grad():
            t_price = torch.tensor(X_price, dtype=torch.float32).to(self.device)
            t_sent = torch.tensor(X_sent, dtype=torch.float32).to(self.device)
            preds = self.model(t_price, t_sent)
            return preds.cpu().numpy().flatten()

    def save(self, filepath: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'state_dict': self.model.state_dict(),
            'seq_len': self.sequence_length,
            'num_price_features': self.num_price_features,
            'num_sent_features': self.num_sentiment_features
        }, filepath)
        print(f"[Model] Saved checkpoint to {filepath}")

    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        print(f"[Model] Loaded checkpoint from {filepath}")


def build_hybrid_model(sequence_length: int, num_price_features: int, num_sentiment_features: int):
    """Factory helper returning the HybridStockPredictor."""
    return HybridStockPredictor(
        sequence_length=sequence_length,
        num_price_features=num_price_features,
        num_sentiment_features=num_sentiment_features
    )
