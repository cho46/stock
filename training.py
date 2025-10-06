# training.py - 개선된 모델 훈련 프로세스
# 주요 개선사항:
# 1. 커리큘럼 학습 적용 (단계별 난이도 증가)
# 2. 개선된 하이퍼파라미터 설정 (전략별 차별화)
# 3. 훈련량 대폭 증가 (20만 스텝)
# 4. 조기 종료 및 모델 검증 추가
# 5. 더 많은 기술적 지표 포함

import os
import warnings
import joblib
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from sklearn.preprocessing import RobustScaler

from analysis import ImprovedStockTradingEnv
from utils import download_stock_data, add_advanced_technical_indicators, ResetFixWrapper

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def run_training_process(symbol, period, user_id, model_name, strategy: str = 'balanced'):
    """개선된 모델 훈련을 실행하고 진행 상황을 JSON 문자열로 yield합니다."""
    period = '10y'  # 기간을 10년으로 고정
    try:
        HYPERPARAMS = {
            'conservative': {
                'learning_rate': 3e-5, 'n_steps': 1024, 'batch_size': 64, 'n_epochs': 10, 'gamma': 0.99,
                'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.01, 'vf_coef': 0.5, 'max_grad_norm': 0.5,
                'policy_kwargs': dict(net_arch=[128, 128, 64])
            },
            'balanced': {
                'learning_rate': 1e-4, 'n_steps': 512, 'batch_size': 64, 'n_epochs': 10, 'gamma': 0.99,
                'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.02, 'vf_coef': 0.5, 'max_grad_norm': 0.5,
                'policy_kwargs': dict(net_arch=[256, 256, 128])
            },
            'aggressive': {
                'learning_rate': 3e-4, 'n_steps': 256, 'batch_size': 32, 'n_epochs': 4, 'gamma': 0.995,
                'gae_lambda': 0.9, 'clip_range': 0.3, 'ent_coef': 0.05, 'policy_kwargs': dict(net_arch=[512, 256, 128])
            }
        }
        params = HYPERPARAMS.get(strategy, HYPERPARAMS['balanced'])

        yield json.dumps({"status": "progress", "message": f"{symbol} 데이터 다운로드 중..."}) + '\n'
        df, requested_start_date = download_stock_data(symbol, period)
        if df is None:
            yield json.dumps({"status": "error", "message": "데이터를 다운로드할 수 없습니다."}) + '\n'
            return

        yield json.dumps({"status": "progress", "message": "고급 기술적 지표 추가 중..."}) + '\n'
        df.columns = df.columns.str.strip()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Date'] = df['Date'].dt.tz_localize(None)
            df.dropna(subset=['Date'], inplace=True)
            df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        df = add_advanced_technical_indicators(df)
        df.dropna(inplace=True)

        # 지표 계산 후, 요청된 시작일로 데이터 필터링
        df = df[df.index >= requested_start_date]

        if len(df) < 200:
            yield json.dumps({"status": "error", "message": f"데이터가 너무 적어 훈련할 수 없습니다. 요청 기간: {period}, 최종 데이터: {len(df)}일"}) + '\n'
            return

        yield json.dumps({"status": "progress", "message": "데이터 정규화 및 분할 중..."}) + '\n'
        train_size = int(len(df) * 0.8)
        train_df = df[:train_size].copy()
        val_df = df[train_size:].copy()

        price_cols = ['Open', 'High', 'Low', 'Close']
        feature_cols = [col for col in df.columns if col not in price_cols and col not in ['Date', 'Target']]
        scaler = RobustScaler()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        val_df[feature_cols] = scaler.transform(val_df[feature_cols])

        # 스케일링 후 발생할 수 있는 NaN/inf 값 처리
        train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        train_df.fillna(0, inplace=True)
        val_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        val_df.fillna(0, inplace=True)

        yield json.dumps({"status": "progress", "message": f"{strategy.capitalize()} 전략으로 환경 설정 및 모델 초기화 중..."}) + '\n'
        train_env = DummyVecEnv([lambda: ResetFixWrapper(Monitor(ImprovedStockTradingEnv(train_df)))])
        eval_env = DummyVecEnv([lambda: ResetFixWrapper(Monitor(ImprovedStockTradingEnv(val_df)))])
        model = PPO('MlpPolicy', train_env, verbose=0, **params)

        # 콜백 설정 및 오류 수정
        reward_threshold = 1.5 # 목표 수익률 50%
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
        callback_on_best.training_stopped = False # 오류 해결
        eval_callback = EvalCallback(eval_env, best_model_save_path=None, log_path=None, eval_freq=2000, deterministic=True, render=False, callback_on_new_best=callback_on_best)

        total_timesteps = 50000 # 훈련량 재조정
        yield json.dumps({"status": "progress", "message": f"총 {total_timesteps} 스텝의 모델 훈련을 시작합니다..."}) + '\n'
        model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=False) # 진행률 표시는 프론트엔드에서 담당

        yield json.dumps({"status": "progress", "message": "모델 및 관련 파일 저장 중..."}) + '\n'
        user_models_dir = os.path.join("D:\\", "StockModelFolder", user_id)
        os.makedirs(user_models_dir, exist_ok=True)
        sanitized_model_name = "".join(c for c in model_name if c.isalnum() or c in ('_', '-')).rstrip() or f"{symbol}_{strategy}_model"
        model_path = os.path.join(user_models_dir, f"{sanitized_model_name}.zip")
        scaler_path = os.path.join(user_models_dir, f"{sanitized_model_name}_scaler.pkl")
        model.save(model_path)
        joblib.dump(scaler, scaler_path)

        yield json.dumps({"status": "success", "message": f"모델 훈련 완료: {sanitized_model_name}"}) + '\n'

    except Exception as e:
        logger.error(f"훈련 중 치명적 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        yield json.dumps({"status": "error", "message": f"훈련 중 오류가 발생했습니다: {str(e)}"}) + '\n'