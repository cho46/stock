# utils.py - 개선된 데이터 처리 유틸리티
# 주요 개선사항:
# 1. 고급 기술적 지표 추가 (MACD, 볼린저 밴드, ATR, Stochastic 등)
# 2. 데이터 품질 검증 및 이상치 처리
# 3. 더 안정적인 데이터 다운로드
# 4. 거래량 기반 지표 추가

import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import warnings
import gymnasium as gym
from azure.storage.blob import BlobServiceClient
import os

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def download_stock_data(symbol, period='1y', start_date=None, end_date=None):
    """개선된 주식 데이터 다운로드 함수 (버퍼 추가)"""
    try:
        end = datetime.now()
        # 기간을 날짜로 변환
        if start_date and end_date:
            start = pd.to_datetime(start_date)
        else:
            num = int("".join(filter(str.isdigit, period)) or "1")
            if 'y' in period:
                start = end - timedelta(days=num * 365)
            elif 'mo' in period:
                start = end - timedelta(days=num * 30)
            else: # 기본값
                start = end - timedelta(days=365)

        # 기술적 지표 계산을 위한 250일 버퍼 추가
        buffer_start = start - timedelta(days=250)

        # yfinance를 통한 데이터 다운로드
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=buffer_start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))

        if df.empty:
            logger.error(f"데이터를 찾을 수 없습니다: {symbol}")
            return None, None

        # 데이터 정제
        df = df.reset_index()
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df = validate_and_clean_data(df)

        if len(df) < 30:
            logger.error(f"유효한 데이터가 너무 적습니다: {len(df)}일")
            return None, None

        logger.info(f"데이터 다운로드 및 전처리 완료: {symbol}, {len(df)}일 데이터")
        # 전체 버퍼 데이터프레임과, 사용자가 원했던 실제 시작 날짜를 함께 반환
        return df, start

    except Exception as e:
        logger.error(f"데이터 다운로드 오류 ({symbol}): {e}")
        return None, None


def validate_and_clean_data(df):
    """데이터 품질 검증 및 정제"""
    try:
        # 결측값 처리
        df = df.dropna()

        # 가격 데이터 유효성 검사
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            # 음수 가격 제거
            df = df[df[col] > 0]

            # 이상치 제거 (3 표준편차 초과)
            mean = df[col].mean()
            std = df[col].std()
            df = df[abs(df[col] - mean) <= 3 * std]

        # OHLC 논리적 일관성 검사
        df = df[
            (df['High'] >= df['Low']) &
            (df['High'] >= df['Open']) &
            (df['High'] >= df['Close']) &
            (df['Low'] <= df['Open']) &
            (df['Low'] <= df['Close'])
            ]

        # 거래량 검증 (음수 제거)
        if 'Volume' in df.columns:
            df = df[df['Volume'] >= 0]

        # 인덱스 재설정
        df = df.reset_index(drop=True)

        logger.info(f"데이터 정제 완료: {len(df)}개 행")
        return df

    except Exception as e:
        logger.error(f"데이터 정제 중 오류: {e}")
        return df


def add_technical_indicators(df):
    """기존 기술적 지표 추가 (호환성 유지)"""
    try:
        df = df.copy()

        # 이동평균
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()

        # RSI
        df['RSI_14'] = calculate_rsi(df['Close'], 14)

        # 모멘텀
        df['Momentum_1'] = df['Close'].pct_change(1)
        df['Momentum_5'] = df['Close'].pct_change(5)

        # 기본 MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

        return df

    except Exception as e:
        logger.error(f"기술적 지표 추가 중 오류: {e}")
        return df


def add_advanced_technical_indicators(df):
    """고급 기술적 지표 추가"""
    try:
        df = df.copy()

        # 기본 지표들 먼저 추가
        df = add_technical_indicators(df)

        # 1. 확장된 이동평균
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['EMA_50'] = df['Close'].ewm(span=50).mean()

        # 2. 볼린저 밴드
        bb_period = 20
        bb_std = 2
        df['BB_Middle'] = df['Close'].rolling(window=bb_period).mean()
        bb_std_dev = df['Close'].rolling(window=bb_period).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std_dev * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std_dev * bb_std)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        # 3. ATR (Average True Range) - 변동성 지표
        df['ATR'] = calculate_atr(df, 14)

        # 4. 스토캐스틱
        df['Stoch_K'], df['Stoch_D'] = calculate_stochastic(df, 14, 3)

        # 5. CCI (Commodity Channel Index)
        df['CCI'] = calculate_cci(df, 20)

        # 6. 윌리엄스 %R
        df['Williams_R'] = calculate_williams_r(df, 14)

        # 7. 거래량 지표
        if 'Volume' in df.columns:
            # 거래량 이동평균
            df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']

            # OBV (On-Balance Volume)
            df['OBV'] = calculate_obv(df)

            # VWAP (Volume Weighted Average Price)
            df['VWAP'] = calculate_vwap(df, 20)

        # 8. 추가 모멘텀 지표
        df['ROC_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        df['ROC_20'] = ((df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20)) * 100

        # 9. 가격 채널
        df['Price_Channel_High'] = df['High'].rolling(window=20).max()
        df['Price_Channel_Low'] = df['Low'].rolling(window=20).min()
        df['Price_Channel_Position'] = (df['Close'] - df['Price_Channel_Low']) / (
                    df['Price_Channel_High'] - df['Price_Channel_Low'])

        # 10. 트렌드 강도
        df['ADX'] = calculate_adx(df, 14)

        logger.info("고급 기술적 지표 추가 완료")
        
        # 최종 데이터 정제 (inf, NaN 처리)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)

        return df

    except Exception as e:
        logger.error(f"고급 기술적 지표 추가 중 오류: {e}")
        return df


def calculate_rsi(prices, period=14):
    """RSI 계산"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_atr(df, period=14):
    """ATR (Average True Range) 계산"""
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    return true_range.rolling(window=period).mean()


def calculate_stochastic(df, k_period=14, d_period=3):
    """스토캐스틱 오실레이터 계산"""
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()

    k_percent = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    d_percent = k_percent.rolling(window=d_period).mean()

    return k_percent, d_percent


def calculate_cci(df, period=20):
    """CCI (Commodity Channel Index) 계산"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    sma = typical_price.rolling(window=period).mean()
    # pandas 2.0에서 mad()가 제거되어 수동으로 계산
    mad = typical_price.rolling(window=period).apply(lambda x: (x - x.mean()).abs().mean())

    return (typical_price - sma) / (0.015 * mad)


def calculate_williams_r(df, period=14):
    """윌리엄스 %R 계산"""
    high_max = df['High'].rolling(window=period).max()
    low_min = df['Low'].rolling(window=period).min()

    return -100 * ((high_max - df['Close']) / (high_max - low_min))


def calculate_obv(df):
    """OBV (On-Balance Volume) 계산"""
    obv = pd.Series(index=df.index, dtype=float)
    obv.iloc[0] = df['Volume'].iloc[0]

    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] + df['Volume'].iloc[i]
        elif df['Close'].iloc[i] < df['Close'].iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] - df['Volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i - 1]

    return obv


def calculate_vwap(df, period=20):
    """VWAP (Volume Weighted Average Price) 계산"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).rolling(window=period).sum() / df['Volume'].rolling(window=period).sum()
    return vwap


def calculate_adx(df, period=14):
    """ADX (Average Directional Index) 계산"""
    # True Range 계산
    tr = calculate_atr(df, 1)  # 1일 TR

    # Directional Movement 계산
    dm_plus = np.where(
        (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
        np.maximum(df['High'] - df['High'].shift(1), 0),
        0
    )

    dm_minus = np.where(
        (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
        np.maximum(df['Low'].shift(1) - df['Low'], 0),
        0
    )

    # Smoothed values
    tr_smooth = pd.Series(tr).rolling(window=period).mean()
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=period).mean()
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=period).mean()

    # Directional Indicators
    di_plus = 100 * (dm_plus_smooth / tr_smooth)
    di_minus = 100 * (dm_minus_smooth / tr_smooth)

    # DX and ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=period).mean()

    return adx


def calculate_macd(df, fast=12, slow=26, signal=9):
    """MACD 계산 (상세 버전)"""
    ema_fast = df['Close'].ewm(span=fast).mean()
    ema_slow = df['Close'].ewm(span=slow).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def detect_support_resistance(df, window=20):
    """지지/저항선 감지"""
    try:
        # 고점과 저점 찾기
        highs = df['High'].rolling(window=window, center=True).max()
        lows = df['Low'].rolling(window=window, center=True).min()

        resistance_levels = df[df['High'] == highs]['High'].unique()
        support_levels = df[df['Low'] == lows]['Low'].unique()

        return sorted(support_levels), sorted(resistance_levels, reverse=True)
    except Exception as e:
        logger.error(f"지지/저항선 계산 오류: {e}")
        return [], []


def add_market_regime_indicators(df):
    """시장 상황 판단 지표 추가"""
    try:
        df = df.copy()

        # 트렌드 방향성
        df['Trend_20'] = np.where(df['Close'] > df['SMA_20'], 1, -1)
        df['Trend_50'] = np.where(df['Close'] > df['SMA_50'], 1, -1) if 'SMA_50' in df.columns else 0

        # 변동성 체계 (고/저 변동성)
        df['Volatility_Regime'] = np.where(df['ATR'] > df['ATR'].rolling(50).mean(), 1, 0) if 'ATR' in df.columns else 0

        # 거래량 체계
        if 'Volume' in df.columns:
            df['Volume_Regime'] = np.where(df['Volume'] > df['Volume'].rolling(20).mean(), 1, 0)
        
        return df

    except Exception as e:
        logger.error(f"시장 상황 지표 추가 오류: {e}")
        return df


class ResetFixWrapper(gym.Wrapper):
    """
    A wrapper to fix an issue where the environment's reset method returns more values than expected.
    This wrapper ensures that reset() returns only the first two values (obs, info).
    """
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        reset_returns = self.env.reset(**kwargs)
        return reset_returns[0], reset_returns[1]

    def step(self, action):
        return self.env.step(action)

# --- Azure Blob Storage Functions ---

def get_blob_service_client():
    """환경 변수에서 연결 문자열을 가져와 BlobServiceClient를 생성합니다."""
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    if not connect_str:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING 환경 변수를 설정해야 합니다.")
    return BlobServiceClient.from_connection_string(connect_str)

def upload_to_blob_storage(local_file_path, blob_name):
    """로컬 파일을 Azure Blob Storage에 업로드합니다."""
    try:
        blob_service_client = get_blob_service_client()
        container_name = "models"
        
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        
        with open(local_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        logger.info(f"'{local_file_path}'를 '{blob_name}'으로 성공적으로 업로드했습니다.")
        return True
    except Exception as e:
        logger.error(f"Blob Storage 업로드 실패: {e}")
        return False

def download_from_blob_storage(blob_name, local_file_path):
    """Azure Blob Storage에서 파일을 다운로드합니다."""
    try:
        blob_service_client = get_blob_service_client()
        container_name = "models"
            
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        
        with open(local_file_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        logger.info(f"'{blob_name}'을 '{local_file_path}'으로 성공적으로 다운로드했습니다.")
        return True
    except Exception as e:
        logger.error(f"Blob Storage 다운로드 실패: {e}")
        return False

def delete_blob(blob_name):
    """Azure Blob Storage에서 파일을 삭제합니다."""
    try:
        blob_service_client = get_blob_service_client()
        container_name = "models"
        
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        blob_client.delete_blob()
        logger.info(f"'{blob_name}'을 성공적으로 삭제했습니다.")
        return True
    except Exception as e:
        logger.error(f"Blob 삭제 실패: {e}")
        return False