import argparse
import yfinance as yf
import joblib
import os
from analyzer import StockAnalyzer


def train_and_save(symbol, period='5y', model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period)
    if data is None or data.empty:
        print(f"No data for {symbol}")
        return

    analyzer = StockAnalyzer()
    data = analyzer.calculate_technical_indicators(data)
    model_info = analyzer.train_prediction_model(data)

    if model_info is None:
        print("Insufficient data to train model")
        return

    # Save model and scaler
    model_path = os.path.join(model_dir, f"{symbol}_model.joblib")
    joblib.dump({'model': analyzer.model, 'scaler': analyzer.scaler, 'feature_cols': model_info['feature_cols']}, model_path)
    print(f"Saved model to {model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', required=True, help='Ticker symbol (e.g., AAPL)')
    parser.add_argument('--period', default='5y', help='History period (default: 5y)')
    parser.add_argument('--model-dir', default='models', help='Directory to save model')
    args = parser.parse_args()

    train_and_save(args.symbol.upper(), args.period, args.model_dir)
