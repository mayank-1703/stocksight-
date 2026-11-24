import unittest
import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient

from src.data.preprocessing import DataPreprocessor
from src.models.hybrid_model import TorchHybridModel
from src.evaluation.evaluate import evaluate_predictions, backtest_trading_strategy
from src.api.main import app

class TestStockSightPipeline(unittest.TestCase):
    def test_feature_engineering_and_sequences(self):
        dates = pd.bdate_range(end='2026-11-20', periods=50)
        prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.001, 0.01, size=50)))
        df = pd.DataFrame({
            'timestamp': dates,
            'Open': prices * 0.99,
            'High': prices * 1.01,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, size=50),
            'sentiment_pos': np.random.uniform(0.1, 0.4, size=50),
            'sentiment_neg': np.random.uniform(0.0, 0.2, size=50),
            'sentiment_neu': np.random.uniform(0.4, 0.7, size=50),
            'sentiment_score': np.random.uniform(-0.1, 0.3, size=50)
        })

        preprocessor = DataPreprocessor(sequence_length=5)
        merged_df = preprocessor.align_and_merge(df, df[['timestamp', 'sentiment_pos', 'sentiment_neg', 'sentiment_neu', 'sentiment_score']])
        dataset = preprocessor.prepare_dataset(merged_df, train_split=0.8)

        self.assertEqual(dataset['X_price_train'].shape[1], 5)
        self.assertEqual(dataset['X_price_train'].shape[2], 5)
        self.assertEqual(dataset['X_sent_train'].shape[1], 4)
        self.assertEqual(len(dataset['y_train']), len(dataset['X_price_train']))

    def test_hybrid_model_forward(self):
        model = TorchHybridModel(
            sequence_length=10,
            num_price_features=5,
            num_sentiment_features=4,
            hidden_dim=32
        )
        x_price = torch.randn(8, 10, 5)
        x_sent = torch.randn(8, 4)
        out = model(x_price, x_sent)
        self.assertEqual(out.shape, torch.Size([8, 1]))

    def test_evaluation_metrics(self):
        y_true = np.array([102.0, 105.0, 103.0, 107.0])
        y_pred = np.array([101.5, 104.5, 103.5, 106.5])
        prev_close = np.array([100.0, 102.0, 105.0, 103.0])

        metrics = evaluate_predictions(y_true, y_pred, prev_close)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('mape', metrics)
        self.assertIn('directional_accuracy', metrics)
        self.assertLess(metrics['mape'], 2.0)

    def test_api_health(self):
        client = TestClient(app)
        response = client.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'healthy')

if __name__ == '__main__':
    unittest.main()
