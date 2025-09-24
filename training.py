import os
import warnings
import joblib
import logging
import json
import pandas as pd
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from sklearn.preprocessing import RobustScaler

from analysis import OptimizedStockTradingEnv
from utils import download_stock_data, add_technical_indicators

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

def run_training_process(symbol, period):
    """모델 훈련을 실행하고 진행 상황을 JSON 문자열로 yield합니다."""
    try:
        yield json.dumps({"status": "progress", "message": f"{symbol} 데이터 다운로드 중..."}) + '\n'
        df = download_stock_data(symbol, period)
        if df is None: 
            yield json.dumps({"status": "error", "message": "데이터를 다운로드할 수 없습니다."}) + '\n'
            return

        yield json.dumps({"status": "progress", "message": "기술적 지표 추가 중..."}) + '\n'
        df.columns = df.columns.str.strip()
        if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        df = add_technical_indicators(df)
        df.dropna(inplace=True)
        if len(df) < 100: 
            yield json.dumps({"status": "error", "message": "데이터가 너무 적어 훈련할 수 없습니다."}) + '\n'
            return

        yield json.dumps({"status": "progress", "message": "데이터 정규화 중..."}) + '\n'
        train_size = int(len(df) * 0.8)
        train_df = df[:train_size].copy()
        price_cols = ['Open', 'High', 'Low', 'Close']
        feature_cols = [col for col in df.columns if col not in price_cols and col not in ['Date', 'Target']]
        scaler = RobustScaler()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])

        yield json.dumps({"status": "progress", "message": "PPO 모델 학습 중... (약 1~2분 소요)"}) + '\n'
        def make_env():
            env = OptimizedStockTradingEnv(train_df)
            return Monitor(env)
        train_env = DummyVecEnv([make_env])

        model = PPO('MlpPolicy', train_env, verbose=0, learning_rate=0.0001, n_steps=256, batch_size=32, n_epochs=4, gamma=0.95, policy_kwargs=dict(net_arch=[128, 128]))
        model.learn(total_timesteps=30000)

        yield json.dumps({"status": "progress", "message": "모델 및 스케일러 저장 중..."}) + '\n'
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{symbol}_ppo_{timestamp}.zip"
        scaler_filename = f"{symbol}_ppo_{timestamp}_scaler.pkl"
        model_path = os.path.join(models_dir, model_filename)
        scaler_path = os.path.join(models_dir, scaler_filename)
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        
        yield json.dumps({"status": "success", "message": f"{symbol} 모델 훈련 완료!", "saved_model": model_filename}) + '\n'

    except Exception as e:
        logger.error(f"훈련 중 오류 발생: {e}")
        yield json.dumps({"status": "error", "message": f"훈련 중 오류가 발생했습니다: {str(e)}"}) + '\n'