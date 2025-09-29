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
from utils import download_stock_data, add_advanced_technical_indicators

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def run_training_process(symbol, period, user_id, model_name, strategy: str = 'balanced'):
    """개선된 모델 훈련을 실행하고 진행 상황을 JSON 문자열로 yield합니다."""
    try:
        # 개선된 하이퍼파라미터 프리셋 정의
        HYPERPARAMS = {
            'conservative': {
                'learning_rate': 3e-5,
                'n_steps': 1024,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'policy_kwargs': dict(net_arch=[128, 128, 64])
            },
            'balanced': {
                'learning_rate': 1e-4,
                'n_steps': 512,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.02,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'policy_kwargs': dict(net_arch=[256, 256, 128])
            },
            'aggressive': {
                'learning_rate': 3e-4,
                'n_steps': 256,
                'batch_size': 32,
                'n_epochs': 4,
                'gamma': 0.995,
                'gae_lambda': 0.9,
                'clip_range': 0.3,
                'ent_coef': 0.05,
                'vf_coef': 0.5,
                'max_grad_norm': 1.0,
                'policy_kwargs': dict(net_arch=[512, 256, 128])
            }
        }
        params = HYPERPARAMS.get(strategy, HYPERPARAMS['balanced'])

        yield json.dumps({"status": "progress", "message": f"{symbol} 데이터 다운로드 중..."}) + '\n'

        # 데이터 다운로드 및 전처리
        df = download_stock_data(symbol, period)
        if df is None:
            yield json.dumps({"status": "error", "message": "데이터를 다운로드할 수 없습니다."}) + '\n'
            return

        yield json.dumps({"status": "progress", "message": "고급 기술적 지표 추가 중..."}) + '\n'

        # 데이터 정제
        df.columns = df.columns.str.strip()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.dropna(subset=['Date'], inplace=True)
            df.set_index('Date', inplace=True)

        df.sort_index(inplace=True)

        # 고급 기술적 지표 추가
        df = add_advanced_technical_indicators(df)
        df.dropna(inplace=True)

        if len(df) < 200:  # 최소 데이터 요구량 증가
            yield json.dumps({"status": "error", "message": "데이터가 너무 적어 훈련할 수 없습니다. 최소 200일 데이터가 필요합니다."}) + '\n'
            return

        yield json.dumps({"status": "progress", "message": "데이터 정규화 및 분할 중..."}) + '\n'

        # 훈련/검증 데이터 분할 (80/20)
        train_size = int(len(df) * 0.8)
        train_df = df[:train_size].copy()
        val_df = df[train_size:].copy()

        # 스케일링
        price_cols = ['Open', 'High', 'Low', 'Close']
        feature_cols = [col for col in df.columns if col not in price_cols and col not in ['Date', 'Target']]

        scaler = RobustScaler()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        val_df[feature_cols] = scaler.transform(val_df[feature_cols])

        yield json.dumps({"status": "progress", "message": f"{strategy.capitalize()} 전략으로 환경 설정 중..."}) + '\n'

        # 훈련 환경 생성
        def make_train_env():
            env = ImprovedStockTradingEnv(train_df, initial_balance=10000)
            return Monitor(env)

        # 검증 환경 생성
        def make_eval_env():
            env = ImprovedStockTradingEnv(val_df, initial_balance=10000)
            return Monitor(env)

        train_env = DummyVecEnv([make_train_env])
        eval_env = DummyVecEnv([make_eval_env])

        yield json.dumps({"status": "progress", "message": "PPO 모델 초기화 중..."}) + '\n'

        # PPO 모델 생성
        model = PPO('MlpPolicy', train_env, verbose=0, **params)

        # 콜백 설정 (조기 종료 및 모델 저장)
        reward_threshold = 0.1  # 10% 수익률 달성 시 조기 종료
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=None,
            log_path=None,
            eval_freq=5000,
            deterministic=True,
            render=False,
            callback_on_new_best=callback_on_best
        )

        # 커리큘럼 학습 - 3단계로 나누어 점진적으로 학습
        total_timesteps = 200000  # 총 20만 스텝
        curriculum_phases = [
            {"name": "기초 학습", "timesteps": 50000, "description": "기본적인 거래 패턴 학습"},
            {"name": "고급 학습", "timesteps": 100000, "description": "복잡한 시장 상황 대응 학습"},
            {"name": "마스터 학습", "timesteps": 50000, "description": "최적화 및 미세 조정"}
        ]

        completed_timesteps = 0
        for phase_idx, phase in enumerate(curriculum_phases):
            yield json.dumps({
                "status": "progress",
                "message": f"Phase {phase_idx + 1}/3: {phase['name']} 시작 - {phase['description']}"
            }) + '\n'

            phase_timesteps = phase["timesteps"]
            steps_per_update = phase_timesteps // 10  # 각 페이즈를 10단계로 분할

            for step in range(10):
                try:
                    # 학습 실행
                    model.learn(
                        total_timesteps=steps_per_update,
                        reset_num_timesteps=False,
                        callback=eval_callback
                    )

                    completed_timesteps += steps_per_update
                    progress = int((completed_timesteps / total_timesteps) * 100)

                    # 중간 성과 평가
                    if step % 3 == 0:  # 3단계마다 평가
                        obs = eval_env.reset()
                        total_reward = 0
                        for _ in range(100):  # 100스텝 테스트
                            action, _ = model.predict(obs, deterministic=True)
                            obs, reward, done, info = eval_env.step(action)
                            total_reward += reward[0]
                            if done[0]:
                                break

                        avg_reward = total_reward / 100

                        yield json.dumps({
                            "status": "progress_update",
                            "percentage": progress,
                            "message": f"Phase {phase_idx + 1} - 진행률: {progress}%, 평가 점수: {avg_reward:.4f}"
                        }) + '\n'
                    else:
                        yield json.dumps({
                            "status": "progress_update",
                            "percentage": progress,
                            "message": f"Phase {phase_idx + 1} - 진행률: {progress}%"
                        }) + '\n'

                    # 조기 종료 확인
                    if callback_on_best.training_stopped:
                        yield json.dumps({
                            "status": "progress",
                            "message": f"목표 성능 달성! 조기 종료됩니다. (수익률 {reward_threshold * 100}% 달성)"
                        }) + '\n'
                        break

                except Exception as e:
                    logger.warning(f"학습 중 일시적 오류: {e}, 계속 진행합니다.")
                    continue

            if callback_on_best.training_stopped:
                break

            yield json.dumps({
                "status": "progress",
                "message": f"Phase {phase_idx + 1} 완료!"
            }) + '\n'

        yield json.dumps({"status": "progress", "message": "최종 모델 검증 중..."}) + '\n'

        # 최종 성과 테스트
        obs = eval_env.reset()
        final_portfolio_value = 10000
        test_rewards = []

        for _ in range(len(val_df) - 50):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            test_rewards.append(reward[0])
            if done[0]:
                final_portfolio_value = info[0].get('net_worth', 10000)
                break

        final_return = (final_portfolio_value - 10000) / 10000
        avg_reward = np.mean(test_rewards) if test_rewards else 0

        yield json.dumps({
            "status": "progress",
            "message": f"검증 완료 - 예상 수익률: {final_return * 100:.2f}%, 평균 보상: {avg_reward:.4f}"
        }) + '\n'

        yield json.dumps({"status": "progress", "message": "모델 및 스케일러 저장 중..."}) + '\n'

        # 모델 저장
        user_models_dir = os.path.join("models", user_id)
        os.makedirs(user_models_dir, exist_ok=True)

        # 파일명 정리
        sanitized_model_name = "".join(c for c in model_name if c.isalnum() or c in ('_', '-')).rstrip()
        if not sanitized_model_name:
            sanitized_model_name = f"{symbol}_{strategy}_model"

        model_filename = f"{sanitized_model_name}.zip"
        scaler_filename = f"{sanitized_model_name}_scaler.pkl"
        model_path = os.path.join(user_models_dir, model_filename)
        scaler_path = os.path.join(user_models_dir, scaler_filename)

        # 모델과 스케일러 저장
        model.save(model_path)
        joblib.dump(scaler, scaler_path)

        # 모델 메타데이터 저장
        metadata = {
            'symbol': symbol,
            'period': period,
            'strategy': strategy,
            'training_date': datetime.now().isoformat(),
            'final_return': final_return,
            'avg_reward': avg_reward,
            'total_timesteps': completed_timesteps,
            'model_params': params
        }

        metadata_filename = f"{sanitized_model_name}_metadata.json"
        metadata_path = os.path.join(user_models_dir, metadata_filename)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        yield json.dumps({
            "status": "success",
            "message": f"🎉 {symbol} 모델 훈련 완료! 예상 수익률: {final_return * 100:.2f}%",
            "saved_model": model_filename,
            "final_return": final_return,
            "total_trades_estimate": len(test_rewards)
        }) + '\n'

    except Exception as e:
        logger.error(f"훈련 중 치명적 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        yield json.dumps({"status": "error", "message": f"훈련 중 오류가 발생했습니다: {str(e)}"}) + '\n'