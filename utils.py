
import pandas as pd
import numpy as np
import yfinance as yf
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from typing import Optional, Dict

logger = logging.getLogger(__name__)

def download_stock_data(symbol: str, period: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
    """주식 데이터를 yfinance로부터 다운로드합니다."""
    try:
        ticker = yf.Ticker(symbol)
        if start_date and end_date:
            df = ticker.history(start=start_date, end=end_date)
        else:
            df = ticker.history(period=period)

        if df.empty:
            logger.warning(f"{symbol}에 대한 데이터를 다운로드할 수 없습니다.")
            return None

        df.reset_index(inplace=True)
        df.columns = [col.replace(' ', '_') for col in df.columns]
        return df
    except Exception as e:
        logger.error(f"데이터 다운로드 중 오류 발생 ({symbol}): {e}")
        return None

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """데이터프레임에 기술적 지표를 추가합니다."""
    df = df.copy()
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"필수 컬럼 {col}이 없습니다.")

    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    try:
        for window in [5, 10, 20, 50]:
            if len(df) > window:
                df[f'SMA_{window}'] = df['Close'].rolling(window=window, min_periods=1).mean()
                df[f'EMA_{window}'] = df['Close'].ewm(span=window, min_periods=1).mean()

        if len(df) > 26:
            df['EMA_12'] = df['Close'].ewm(span=12, min_periods=1).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, min_periods=1).mean()
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_signal'] = df['MACD'].ewm(span=9, min_periods=1).mean()

        for period in [14, 21]:
            if len(df) > period:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
                rs = gain / (loss + 1e-8)
                df[f'RSI_{period}'] = 100 - (100 / (1 + rs))

        for window in [10, 20]:
            if len(df) > window:
                sma = df['Close'].rolling(window=window, min_periods=1).mean()
                std = df['Close'].rolling(window=window, min_periods=1).std()
                df[f'BB_upper_{window}'] = sma + (std * 2)
                df[f'BB_lower_{window}'] = sma - (std * 2)

        df['Volume_SMA_10'] = df['Volume'].rolling(window=10, min_periods=1).mean()
        df['Volume_ratio'] = df['Volume'] / (df['Volume_SMA_10'] + 1)
        
        returns = df['Close'].pct_change().fillna(0)
        df['Volatility_10'] = returns.rolling(window=10, min_periods=1).std()
        
        df['Momentum_1'] = df['Close'] / df['Close'].shift(1) - 1
        df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1

    except Exception as e:
        logger.error(f"기술적 지표 계산 중 오류: {e}")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True)
    return df
