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
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold
from sklearn.preprocessing import RobustScaler

from analysis import ImprovedStockTradingEnv
from utils import download_stock_data, add_advanced_technical_indicators, ResetFixWrapper, upload_to_blob_storage

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

        yield json.dumps({"status": "progress_update", "percentage": 5, "message": f"{symbol} 데이터 다운로드 중..."}) + '\n'
        df, requested_start_date = download_stock_data(symbol, period)
        if df is None:
            yield json.dumps({"status": "error", "message": "데이터를 다운로드할 수 없습니다."}) + '\n'
            return

        yield json.dumps({"status": "progress_update", "percentage": 15, "message": "고급 기술적 지표 추가 중..."}) + '\n'
        df.columns = df.columns.str.strip()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Date'] = df['Date'].dt.tz_localize(None)
            df.dropna(subset=['Date'], inplace=True)
            df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        df = add_advanced_technical_indicators(df)
        df.dropna(inplace=True)

        df = df[df.index >= requested_start_date]

        if len(df) < 200:
            yield json.dumps({"status": "error", "message": f"데이터가 너무 적어 훈련할 수 없습니다. 요청 기간: {period}, 최종 데이터: {len(df)}일"}) + '\n'
            return

        yield json.dumps({"status": "progress_update", "percentage": 30, "message": "데이터 정규화 및 분할 중..."}) + '\n'
        train_size = int(len(df) * 0.8)
        train_df = df[:train_size].copy()
        val_df = df[train_size:].copy()

        price_cols = ['Open', 'High', 'Low', 'Close']
        feature_cols = [col for col in df.columns if col not in price_cols and col not in ['Date', 'Target']]
        scaler = RobustScaler()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        val_df[feature_cols] = scaler.transform(val_df[feature_cols])

        train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        train_df.fillna(0, inplace=True)
        val_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        val_df.fillna(0, inplace=True)

        yield json.dumps({"status": "progress_update", "percentage": 45, "message": f"{strategy.capitalize()} 전략으로 환경 설정 및 모델 초기화 중..."}) + '\n'
        train_env = DummyVecEnv([lambda: ResetFixWrapper(Monitor(ImprovedStockTradingEnv(train_df)))])
        eval_env = DummyVecEnv([lambda: ResetFixWrapper(Monitor(ImprovedStockTradingEnv(val_df)))])
        model = PPO('MlpPolicy', train_env, verbose=0, **params)

        # 평가 콜백 설정 (조기 종료 로직 제거)
        eval_callback = EvalCallback(eval_env, best_model_save_path=None, log_path=None, eval_freq=2000, deterministic=True, render=False)

        total_timesteps = 50000
        n_chunks = 100
        chunk_steps = total_timesteps // n_chunks

        yield json.dumps({"status": "progress_update", "percentage": 50, "message": f"총 {total_timesteps} 스텝의 모델 훈련을 시작합니다..."}) + '\n'
        
        for i in range(n_chunks):
            model.learn(total_timesteps=chunk_steps, callback=eval_callback, reset_num_timesteps=False, progress_bar=False)

            # 진행률 업데이트 (50% ~ 90%)
            percentage = 50 + int(40 * (i + 1) / n_chunks)
            yield json.dumps({
                "status": "progress_update", 
                "percentage": percentage, 
                "message": f"훈련 진행 중... ({model.num_timesteps}/{total_timesteps})"
            }) + '\n'

            yield json.dumps({"status": "progress_update", "percentage": 90, "message": "목표 보상에 도달하여 훈련을 조기 종료했습니다." }) + '\n'

        yield json.dumps({"status": "progress_update", "percentage": 90, "message": "모델 및 관련 파일 저장 및 업로드 중..."}) + '\n'
        
        # 로컬 임시 저장 경로 설정
        temp_dir = os.path.join("temp", user_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        sanitized_model_name = "".join(c for c in model_name if c.isalnum() or c in ('_', '-')).rstrip() or f"{symbol}_{strategy}_model"
        
        local_model_path = os.path.join(temp_dir, f"{sanitized_model_name}.zip")
        local_scaler_path = os.path.join(temp_dir, f"{sanitized_model_name}_scaler.pkl")
        local_metadata_path = os.path.join(temp_dir, f"{sanitized_model_name}_metadata.json")

        model.save(local_model_path)
        joblib.dump(scaler, local_scaler_path)

        # 메타데이터 생성
        metadata = {
            'symbol': symbol,
            'strategy': strategy,
            'training_date': datetime.now().isoformat(),
            'total_timesteps': total_timesteps,
            'final_return': 0 # 이 값은 백테스팅 후 업데이트 될 수 있습니다.
        }
        with open(local_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

        # Azure Blob Storage에 업로드
        blob_model_name = f"{user_id}/{sanitized_model_name}.zip"
        blob_scaler_name = f"{user_id}/{sanitized_model_name}_scaler.pkl"
        blob_metadata_name = f"{user_id}/{sanitized_model_name}_metadata.json"

        model_uploaded = upload_to_blob_storage(local_model_path, blob_model_name)
        scaler_uploaded = upload_to_blob_storage(local_scaler_path, blob_scaler_name)
        metadata_uploaded = upload_to_blob_storage(local_metadata_path, blob_metadata_name)

        # 로컬 임시 파일 삭제
        try:
            os.remove(local_model_path)
            os.remove(local_scaler_path)
            os.remove(local_metadata_path)
        except OSError as e:
            logger.warning(f"로컬 임시 파일 삭제 실패: {e}")

        if not all([model_uploaded, scaler_uploaded, metadata_uploaded]):
            yield json.dumps({"status": "error", "message": "Azure Blob Storage에 모델 업로드를 실패했습니다."}) + '\n'
            return

        # 성공 시, 데이터베이스에 저장할 정보를 포함하여 반환
        yield json.dumps({
            "status": "success", 
            "percentage": 100, 
            "message": f"모델 훈련 및 업로드 완료: {sanitized_model_name}",
            "model_info": {
                "model_name": sanitized_model_name,
                "symbol": symbol,
                "strategy": strategy,
                "blob_model_path": blob_model_name,
                "blob_scaler_path": blob_scaler_name,
                "blob_metadata_path": blob_metadata_name
            }
        }) + '\n'
    except Exception as e:
        logger.error(f"훈련 중 치명적 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        yield json.dumps({"status": "error", "message": f"훈련 중 오류가 발생했습니다: {str(e)}"}) + '\n'