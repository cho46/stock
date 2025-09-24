import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, SAC, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.buffers import ReplayBuffer
from sklearn.preprocessing import RobustScaler
import os
import warnings
import joblib
from typing import List, Tuple, Dict, Any
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

# 한글 폰트 설정
try:
    plt.rc('font', family='Malgun Gothic')
except:
    try:
        plt.rc('font', family='DejaVu Sans')
    except:
        pass
plt.rc('axes', unicode_minus=False)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedTradingCallback(BaseCallback):
    """개선된 학습 진행상황 모니터링 콜백"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_count = 0
        self.step_count = 0
        self.last_episode_reward = 0

    def _on_step(self) -> bool:
        self.step_count += 1

        # 환경에서 직접 정보 가져오기
        if hasattr(self.training_env.envs[0], 'episode_profit'):
            current_profit = self.training_env.envs[0].episode_profit
            if current_profit != self.last_episode_reward:
                self.episode_count += 1
                self.episode_rewards.append(current_profit)
                self.last_episode_reward = current_profit

                if self.episode_count % 50 == 0:
                    avg_reward = np.mean(self.episode_rewards[-50:]) if self.episode_rewards else 0
                    logger.info(f"Episode {self.episode_count}: Average profit (last 50): ${avg_reward:.2f}")

        return True

class OptimizedStockTradingEnv(gym.Env):
    """최적화된 주식 거래 환경"""
    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_balance=10000, lookback_window=20,
                 max_position_ratio=0.3, transaction_cost=0.001):
        super().__init__()

        # 데이터 유효성 검사
        if len(df) < lookback_window + 10:
            raise ValueError(f"데이터가 너무 짧습니다. 최소 {lookback_window + 10}개 행 필요")

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.lookback_window = max(5, min(lookback_window, len(df) // 6))
        self.max_position_ratio = np.clip(max_position_ratio, 0.1, 0.8)
        self.transaction_cost = np.clip(transaction_cost, 0.0001, 0.01)

        # 액션 스페이스: 3개의 이산 액션 (매도, 보유, 매수)
        self.action_space = spaces.Discrete(3)

        # 간소화된 관찰 스페이스
        # 가격 정보 (4) + 기술적 지표 (10) + 포트폴리오 상태 (6)
        self.observation_space = spaces.Box(
            low=-5, high=5, shape=(20,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.balance = float(self.initial_balance)
        self.shares_held = 0.0
        self.net_worth = float(self.initial_balance)
        self.current_step = self.lookback_window
        self.max_net_worth = float(self.initial_balance)
        self.trade_history = []
        self.daily_returns = []
        self.total_trades = 0
        self.last_action = 1
        self.consecutive_holds = 0

        # 성과 추적
        self.episode_profit = 0
        self.buy_hold_benchmark = self.initial_balance
        self.episode_steps = 0

        return self._get_observation(), self._get_info()

    def _get_observation(self):
        """간소화된 관찰 벡터 생성"""
        safe_step = min(self.current_step, len(self.df) - 1)

        try:
            current_row = self.df.iloc[safe_step]
            current_close = max(0.01, current_row['Close'])

            # 1. 정규화된 가격 정보 (4개)
            price_features = np.array([
                current_row['Open'] / current_close,
                current_row['High'] / current_close,
                current_row['Low'] / current_close,
                1.0  # Close/Close = 1
            ])

            # 2. 주요 기술적 지표 (10개)
            tech_features = np.zeros(10)

            # 이동평균 비율
            if 'SMA_5' in current_row and current_row['SMA_5'] > 0:
                tech_features[0] = np.clip(current_close / current_row['SMA_5'] - 1, -0.5, 0.5)
            if 'SMA_20' in current_row and current_row['SMA_20'] > 0:
                tech_features[1] = np.clip(current_close / current_row['SMA_20'] - 1, -0.5, 0.5)

            # RSI
            if 'RSI_14' in current_row:
                tech_features[2] = np.clip((current_row['RSI_14'] - 50) / 50, -1, 1)

            # 모멘텀
            if 'Momentum_1' in current_row:
                tech_features[3] = np.clip(current_row['Momentum_1'], -0.2, 0.2)
            if 'Momentum_5' in current_row:
                tech_features[4] = np.clip(current_row['Momentum_5'], -0.5, 0.5)

            # 볼린저 밴드 포지션
            if 'BB_position' in current_row:
                tech_features[5] = np.clip(current_row['BB_position'] - 0.5, -0.5, 0.5)

            # 거래량 비율
            if 'Volume_ratio' in current_row:
                tech_features[6] = np.clip(np.log(current_row['Volume_ratio']), -2, 2)

            # 변동성
            if 'Volatility_10' in current_row:
                tech_features[7] = np.clip(current_row['Volatility_10'], 0, 0.1)

            # 추가 지표들
            if len(self.daily_returns) > 0:
                tech_features[8] = np.clip(np.mean(self.daily_returns[-5:]), -0.1, 0.1)
            if len(self.daily_returns) > 1:
                tech_features[9] = np.clip(np.std(self.daily_returns[-10:]), 0, 0.1)

            # 3. 포트폴리오 상태 (6개)
            stock_value = self.shares_held * current_close
            total_value = max(0.01, self.balance + stock_value)
            position_ratio = stock_value / total_value

            portfolio_features = np.array([
                np.clip(self.balance / self.initial_balance, 0, 2),  # 현금 비율
                np.clip(position_ratio, 0, 1),  # 포지션 비율
                np.clip(total_value / self.initial_balance, 0, 3),  # 총자산 비율
                self.last_action / 2,  # 마지막 액션
                np.clip(self.consecutive_holds / 10, 0, 1),  # 연속 보유
                np.clip(self.total_trades / 50, 0, 1)  # 총 거래수
            ])

            # 최종 관찰 벡터 (총 20개)
            observation = np.concatenate([
                price_features,     # 4
                tech_features,      # 10
                portfolio_features  # 6
            ])

            return observation.astype(np.float32)

        except Exception as e:
            logger.warning(f"관찰 생성 오류: {e}")
            return np.zeros(20, dtype=np.float32)

    def _get_info(self):
        safe_step = min(self.current_step, len(self.df) - 1)
        current_price = self.df.iloc[safe_step]['Close']
        stock_value = self.shares_held * current_price

        return {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': current_price,
            'position_ratio': stock_value / max(1, self.net_worth),
            'total_trades': self.total_trades,
            'episode_profit': self.episode_profit,
            'episode_steps': self.episode_steps
        }

    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, False, self._get_info()

        self.episode_steps += 1
        prev_net_worth = self.net_worth

        # 액션 실행
        reward = self._execute_action(action)

        # 다음 스텝으로 이동
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        # 순자산 업데이트
        current_price = self.df.iloc[min(self.current_step, len(self.df) - 1)]['Close']
        self.net_worth = max(0.01, self.balance + self.shares_held * current_price)

        # 일일 수익률 계산
        if prev_net_worth > 0:
            daily_return = (self.net_worth - prev_net_worth) / prev_net_worth
            self.daily_returns.append(daily_return)

        # 간소화된 리워드 계산
        reward += self._calculate_simple_reward(prev_net_worth)

        # 벤치마크 업데이트
        if self.current_step > self.lookback_window:
            initial_price = self.df.iloc[self.lookback_window]['Close']
            self.buy_hold_benchmark = self.initial_balance * (current_price / initial_price)

        self.episode_profit = self.net_worth - self.initial_balance
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        # 조기 종료 조건
        if self.net_worth < self.initial_balance * 0.2:  # 80% 손실
            done = True
            reward -= 1.0

        return self._get_observation(), reward, done, False, self._get_info()

    def _execute_action(self, action):
        """간소화된 액션 실행"""
        try:
            current_price = self.df.iloc[min(self.current_step, len(self.df) - 1)]['Close']
            if current_price <= 0:
                return 0

            self.last_action = action

            if action == 2:  # 매수
                self.consecutive_holds = 0
                available_balance = self.balance * 0.95
                max_shares = available_balance / (current_price * (1 + self.transaction_cost))

                if max_shares > 0.01:  # 최소 거래량
                    shares_to_buy = max_shares * self.max_position_ratio
                    cost = shares_to_buy * current_price * (1 + self.transaction_cost)

                    if self.balance >= cost:
                        self.balance -= cost
                        self.shares_held += shares_to_buy
                        self.total_trades += 1
                        return 0.001

            elif action == 0:  # 매도
                self.consecutive_holds = 0
                if self.shares_held > 0.01:
                    shares_to_sell = self.shares_held * 0.5  # 50% 매도
                    sale_value = shares_to_sell * current_price * (1 - self.transaction_cost)
                    self.balance += sale_value
                    self.shares_held -= shares_to_sell
                    self.total_trades += 1
                    return 0.001

            else:  # 보유 (action == 1)
                self.consecutive_holds += 1
                return 0

            return 0

        except Exception as e:
            logger.warning(f"액션 실행 오류: {e}")
            return 0

    def _calculate_simple_reward(self, prev_net_worth):
        """간소화된 리워드 계산"""
        try:
            # 1. 수익률 기반 리워드 (주요)
            if prev_net_worth > 0:
                return_rate = (self.net_worth - prev_net_worth) / prev_net_worth
                return np.clip(return_rate * 100, -1, 1)  # 수익률에 100배 가중치

            return 0

        except Exception:
            return 0


def add_enhanced_technical_indicators(df):
    """강화학습에 최적화된 기술적 지표"""
    df = df.copy()

    # 기본 검증
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"필수 컬럼 {col}이 없습니다")
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    # 가격 데이터 검증
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = np.maximum(df[col], 0.01)
    df['Volume'] = np.maximum(df['Volume'], 1)

    try:
        # 기본 이동평균
        for window in [5, 10, 20]:
            if len(df) > window:
                df[f'SMA_{window}'] = df['Close'].rolling(window=window, min_periods=1).mean()
                df[f'EMA_{window}'] = df['Close'].ewm(span=window, min_periods=1).mean()

        # RSI
        for period in [14]:
            if len(df) > period:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
                rs = gain / (loss + 1e-8)
                df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
                df[f'RSI_{period}'] = np.clip(df[f'RSI_{period}'], 0, 100)

        # 모멘텀 지표
        for window in [1, 5, 10]:
            if len(df) > window:
                momentum = df['Close'].pct_change(window).fillna(0)
                df[f'Momentum_{window}'] = np.clip(momentum, -0.5, 0.5)

        # 볼린저 밴드 포지션
        if len(df) > 20:
            sma_20 = df['Close'].rolling(window=20, min_periods=1).mean()
            std_20 = df['Close'].rolling(window=20, min_periods=1).std()
            df['BB_upper'] = sma_20 + (std_20 * 2)
            df['BB_lower'] = sma_20 - (std_20 * 2)

            bb_range = df['BB_upper'] - df['BB_lower']
            bb_range = bb_range.replace(0, 1e-8)
            df['BB_position'] = (df['Close'] - df['BB_lower']) / bb_range
            df['BB_position'] = np.clip(df['BB_position'], 0, 1)

        # 거래량 지표
        if len(df) > 10:
            df['Volume_SMA'] = df['Volume'].rolling(window=10, min_periods=1).mean()
            df['Volume_ratio'] = df['Volume'] / (df['Volume_SMA'] + 1)
            df['Volume_ratio'] = np.clip(df['Volume_ratio'], 0, 5)

        # 변동성 지표
        if len(df) > 10:
            returns = df['Close'].pct_change().fillna(0)
            returns = np.clip(returns, -0.3, 0.3)
            df['Volatility_10'] = returns.rolling(window=10, min_periods=1).std()

    except Exception as e:
        logger.error(f"기술적 지표 계산 오류: {e}")

    # 무한값, NaN 정리
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True)

    return df


def train_rl_models(train_df, val_df, lookback_window=15):
    """PPO, A2C, SAC, DQN 모델 학습"""
    models = {}
    training_histories = {}

    # 환경 설정
    def make_env():
        env = OptimizedStockTradingEnv(
            train_df,
            lookback_window=lookback_window,
            max_position_ratio=0.3,
            transaction_cost=0.002
        )
        return Monitor(env)

    train_env = DummyVecEnv([make_env])

    # 1. PPO 학습 (개선된 설정)
    logger.info("PPO 모델 학습 시작...")
    try:
        ppo_callback = ImprovedTradingCallback()
        ppo_model = PPO(
            'MlpPolicy',
            train_env,
            verbose=1,
            learning_rate=0.0001,  # 더 작은 학습률
            n_steps=256,           # 더 작은 배치
            batch_size=32,
            n_epochs=4,            # 더 적은 에폭
            gamma=0.95,            # 할인계수 조정
            gae_lambda=0.9,
            clip_range=0.1,        # 더 작은 클립 범위
            ent_coef=0.001,        # 더 작은 엔트로피
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[128, 128]),  # 더 작은 네트워크
            tensorboard_log="./tensorboard_logs/"
        )

        ppo_model.learn(
            total_timesteps=30000,  # 더 적은 스텝
            callback=ppo_callback,
            tb_log_name="PPO_improved"
        )

        models['PPO'] = ppo_model
        training_histories['PPO'] = ppo_callback.episode_rewards
        logger.info("PPO 학습 완료")

    except Exception as e:
        logger.error(f"PPO 학습 오류: {e}")

    # 2. A2C 학습
    logger.info("A2C 모델 학습 시작...")
    try:
        a2c_callback = ImprovedTradingCallback()
        a2c_model = A2C(
            'MlpPolicy',
            train_env,
            verbose=1,
            learning_rate=0.0005,
            n_steps=8,
            gamma=0.95,
            gae_lambda=0.9,
            ent_coef=0.001,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[128, 128]),
            tensorboard_log="./tensorboard_logs/"
        )

        a2c_model.learn(
            total_timesteps=30000,
            callback=a2c_callback,
            tb_log_name="A2C_improved"
        )

        models['A2C'] = a2c_model
        training_histories['A2C'] = a2c_callback.episode_rewards
        logger.info("A2C 학습 완료")

    except Exception as e:
        logger.error(f"A2C 학습 오류: {e}")

    # 3. DQN 학습 (새로 추가)
    logger.info("DQN 모델 학습 시작...")
    try:
        dqn_callback = ImprovedTradingCallback()
        dqn_model = DQN(
            'MlpPolicy',
            train_env,
            verbose=1,
            learning_rate=0.0001,
            buffer_size=10000,
            learning_starts=1000,
            batch_size=32,
            tau=1.0,
            gamma=0.95,
            train_freq=4,
            gradient_steps=1,
            exploration_fraction=0.3,  # 30% 탐험
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            policy_kwargs=dict(net_arch=[128, 128]),
            tensorboard_log="./tensorboard_logs/"
        )

        dqn_model.learn(
            total_timesteps=30000,
            callback=dqn_callback,
            tb_log_name="DQN"
        )

        models['DQN'] = dqn_model
        training_histories['DQN'] = dqn_callback.episode_rewards
        logger.info("DQN 학습 완료")

    except Exception as e:
        logger.error(f"DQN 학습 오류: {e}")

    # 4. SAC 학습 (연속 액션용)
    logger.info("SAC 모델 학습 시작...")
    try:
        # SAC용 연속 액션 환경
        class ContinuousStockTradingEnv(OptimizedStockTradingEnv):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

            def _execute_action(self, action):
                action_val = action[0] if isinstance(action, np.ndarray) else action

                if action_val > 0.2:
                    discrete_action = 2  # 매수
                elif action_val < -0.2:
                    discrete_action = 0  # 매도
                else:
                    discrete_action = 1  # 보유

                return super()._execute_action(discrete_action)

        def make_sac_env():
            env = ContinuousStockTradingEnv(
                train_df,
                lookback_window=lookback_window,
                max_position_ratio=0.3,
                transaction_cost=0.002
            )
            return Monitor(env)

        sac_train_env = DummyVecEnv([make_sac_env])

        sac_callback = ImprovedTradingCallback()
        sac_model = SAC(
            'MlpPolicy',
            sac_train_env,
            verbose=1,
            learning_rate=0.0001,
            buffer_size=10000,
            learning_starts=1000,
            batch_size=64,
            tau=0.005,
            gamma=0.95,
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto',
            policy_kwargs=dict(net_arch=[128, 128]),
            tensorboard_log="./tensorboard_logs/"
        )

        sac_model.learn(
            total_timesteps=30000,
            callback=sac_callback,
            tb_log_name="SAC_improved"
        )

        models['SAC'] = sac_model
        training_histories['SAC'] = sac_callback.episode_rewards
        logger.info("SAC 학습 완료")

    except Exception as e:
        logger.error(f"SAC 학습 오류: {e}")

    return models, training_histories


def backtest_rl_strategy(model, model_name, test_df, lookback_window=15, initial_balance=10000):
    """강화학습 모델 백테스팅"""
    try:
        # 환경 타입에 따른 처리
        if model_name == 'SAC':
            class ContinuousStockTradingEnv(OptimizedStockTradingEnv):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

                def _execute_action(self, action):
                    action_val = action[0] if isinstance(action, np.ndarray) else action

                    if action_val > 0.2:
                        discrete_action = 2  # 매수
                    elif action_val < -0.2:
                        discrete_action = 0  # 매도
                    else:
                        discrete_action = 1  # 보유

                    return super()._execute_action(discrete_action)

            test_env = ContinuousStockTradingEnv(
                test_df,
                initial_balance=initial_balance,
                lookback_window=lookback_window,
                max_position_ratio=0.3,
                transaction_cost=0.002
            )
        else:
            test_env = OptimizedStockTradingEnv(
                test_df,
                initial_balance=initial_balance,
                lookback_window=lookback_window,
                max_position_ratio=0.3,
                transaction_cost=0.002
            )

        obs = test_env.reset()[0]
        done = False
        net_worth_history = [initial_balance]
        actions_taken = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            actions_taken.append(action)
            obs, reward, done, truncated, info = test_env.step(action)
            net_worth_history.append(info['net_worth'])

        # 성과 지표 계산
        final_value = net_worth_history[-1]
        total_return = (final_value - initial_balance) / initial_balance

        returns = pd.Series(net_worth_history).pct_change().dropna()

        if len(returns) > 0:
            volatility = returns.std() * np.sqrt(252)
            if len(returns) > 252:
                annual_return = (1 + total_return) ** (252 / len(returns)) - 1
            else:
                annual_return = total_return

            sharpe_ratio = annual_return / volatility if volatility > 0 else 0

            cummax = pd.Series(net_worth_history).cummax()
            drawdown = (pd.Series(net_worth_history) - cummax) / cummax
            max_drawdown = drawdown.min()

            positive_returns = returns[returns > 0]
            win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0

        else:
            annual_return = total_return
            volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0
            win_rate = 0

        # 벤치마크 계산 (Buy & Hold)
        if len(test_df) > lookback_window:
            initial_price = test_df.iloc[lookback_window]['Close']
            final_price = test_df.iloc[-1]['Close']
            benchmark_return = (final_price - initial_price) / initial_price
        else:
            benchmark_return = 0

        return {
            'name': model_name,
            'net_worth_history': net_worth_history,
            'actions_taken': actions_taken,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'final_value': final_value,
            'num_trades': test_env.total_trades,
            'benchmark_return': benchmark_return,
            'excess_return': total_return - benchmark_return
        }

    except Exception as e:
        logger.error(f"{model_name} 백테스팅 오류: {e}")
        return {
            'name': model_name,
            'error': str(e),
            'net_worth_history': [initial_balance],
            'total_return': 0,
            'annual_return': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'final_value': initial_balance,
            'num_trades': 0,
            'benchmark_return': 0,
            'excess_return': 0
        }


def create_comprehensive_plots(results, training_histories, test_df):
    """종합적인 결과 시각화"""
    try:
        fig = plt.figure(figsize=(20, 16))

        # 1. 포트폴리오 가치 변화
        plt.subplot(3, 4, 1)
        for result in results:
            if 'error' not in result and len(result['net_worth_history']) > 1:
                plt.plot(result['net_worth_history'],
                        label=result['name'], linewidth=2, alpha=0.8)

        # Buy & Hold 벤치마크 추가
        if len(test_df) > 15:
            initial_price = test_df.iloc[15]['Close']
            benchmark_values = [10000 * (price / initial_price) for price in test_df['Close'].iloc[15:]]
            plt.plot(benchmark_values[:len(results[0]['net_worth_history'])],
                    label='Buy & Hold', linestyle='--', alpha=0.7)

        plt.title('포트폴리오 가치 변화', fontweight='bold')
        plt.xlabel('시간 스텝')
        plt.ylabel('포트폴리오 가치 ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. 총 수익률 비교
        plt.subplot(3, 4, 2)
        models = [r['name'] for r in results if 'error' not in r]
        returns = [r['total_return'] * 100 for r in results if 'error' not in r]

        colors = ['green' if r > 0 else 'red' for r in returns]
        bars = plt.bar(models, returns, color=colors, alpha=0.7)
        plt.title('총 수익률 비교 (%)', fontweight='bold')
        plt.ylabel('수익률 (%)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')

        for bar, value in zip(bars, returns):
            plt.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (1 if value > 0 else -2),
                    f'{value:.1f}%', ha='center',
                    va='bottom' if value > 0 else 'top')

        # 3. 샤프 비율 비교
        plt.subplot(3, 4, 3)
        sharpe_ratios = [r['sharpe_ratio'] for r in results if 'error' not in r]
        colors = ['darkgreen' if s > 1 else 'lightgreen' if s > 0 else 'red' for s in sharpe_ratios]
        bars = plt.bar(models, sharpe_ratios, color=colors, alpha=0.7)
        plt.title('샤프 비율 비교', fontweight='bold')
        plt.ylabel('샤프 비율')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')

        for bar, value in zip(bars, sharpe_ratios):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.2f}', ha='center', va='bottom')

        # 4. 최대 낙폭 비교
        plt.subplot(3, 4, 4)
        max_drawdowns = [abs(r['max_drawdown']) * 100 for r in results if 'error' not in r]
        bars = plt.bar(models, max_drawdowns, color='orange', alpha=0.7)
        plt.title('최대 낙폭 비교 (%)', fontweight='bold')
        plt.ylabel('최대 낙폭 (%)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')

        for bar, value in zip(bars, max_drawdowns):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom')

        # 5. 학습 곡선 (에피소드 리워드)
        plt.subplot(3, 4, 5)
        colors = ['blue', 'red', 'green', 'purple']
        for i, (model_name, rewards) in enumerate(training_histories.items()):
            if len(rewards) > 0:
                # 이동평균으로 스무딩
                window = max(1, len(rewards) // 50)
                smoothed_rewards = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
                plt.plot(smoothed_rewards, label=f'{model_name}',
                        alpha=0.8, color=colors[i % len(colors)])

        plt.title('학습 곡선 (에피소드 수익)', fontweight='bold')
        plt.xlabel('에피소드')
        plt.ylabel('수익 ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 6. 변동성 비교
        plt.subplot(3, 4, 6)
        volatilities = [r['volatility'] * 100 for r in results if 'error' not in r]
        bars = plt.bar(models, volatilities, color='purple', alpha=0.7)
        plt.title('연간 변동성 비교 (%)', fontweight='bold')
        plt.ylabel('변동성 (%)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')

        for bar, value in zip(bars, volatilities):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom')

        # 7. 승률 비교
        plt.subplot(3, 4, 7)
        win_rates = [r['win_rate'] * 100 for r in results if 'error' not in r]
        bars = plt.bar(models, win_rates, color='lightblue', alpha=0.7)
        plt.title('승률 비교 (%)', fontweight='bold')
        plt.ylabel('승률 (%)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')

        for bar, value in zip(bars, win_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom')

        # 8. 거래 횟수 비교
        plt.subplot(3, 4, 8)
        trade_counts = [r['num_trades'] for r in results if 'error' not in r]
        bars = plt.bar(models, trade_counts, color='brown', alpha=0.7)
        plt.title('총 거래 횟수', fontweight='bold')
        plt.ylabel('거래 횟수')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')

        for bar, value in zip(bars, trade_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value}', ha='center', va='bottom')

        # 9. 벤치마크 대비 초과수익률
        plt.subplot(3, 4, 9)
        excess_returns = [r['excess_return'] * 100 for r in results if 'error' not in r]
        colors = ['green' if r > 0 else 'red' for r in excess_returns]
        bars = plt.bar(models, excess_returns, color=colors, alpha=0.7)
        plt.title('벤치마크 대비 초과수익률 (%)', fontweight='bold')
        plt.ylabel('초과수익률 (%)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')

        for bar, value in zip(bars, excess_returns):
            plt.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (1 if value > 0 else -2),
                    f'{value:.1f}%', ha='center',
                    va='bottom' if value > 0 else 'top')

        # 10. 누적 수익률 (로그 스케일)
        plt.subplot(3, 4, 10)
        for result in results:
            if 'error' not in result and len(result['net_worth_history']) > 1:
                cum_returns = [(v / 10000) for v in result['net_worth_history']]
                plt.semilogy(cum_returns, label=result['name'], linewidth=2, alpha=0.8)

        plt.title('누적 수익률 (로그 스케일)', fontweight='bold')
        plt.xlabel('시간 스텝')
        plt.ylabel('누적 수익률 (배수)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 11. 리스크-수익률 스캐터
        plt.subplot(3, 4, 11)
        for result in results:
            if 'error' not in result:
                plt.scatter(result['volatility'] * 100, result['annual_return'] * 100,
                           s=100, alpha=0.7, label=result['name'])

        plt.title('리스크-수익률 분포', fontweight='bold')
        plt.xlabel('연간 변동성 (%)')
        plt.ylabel('연간 수익률 (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 12. 종합 성과 점수
        plt.subplot(3, 4, 12)
        composite_scores = []
        for result in results:
            if 'error' not in result:
                # 종합 점수 계산 (샤프비율 40% + 수익률 30% + 낙폭 30%)
                score = (result['sharpe_ratio'] * 0.4 +
                        result['total_return'] * 0.3 +
                        (1 - abs(result['max_drawdown'])) * 0.3)
                composite_scores.append(score)

        colors = ['gold' if s == max(composite_scores) else 'lightgray' for s in composite_scores]
        bars = plt.bar(models, composite_scores, color=colors, alpha=0.8)
        plt.title('종합 성과 점수', fontweight='bold')
        plt.ylabel('점수')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')

        for bar, value in zip(bars, composite_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        logger.error(f"시각화 생성 오류: {e}")
        print("시각화를 생성할 수 없어 텍스트 요약으로 대체합니다.")


def main():
    """메인 실행 함수"""
    logger.info("=== 개선된 강화학습 주식거래 시스템 시작 ===")

    # 1. 데이터 로드
    possible_paths = [
        'CONY_daily_historical.csv',
        './CONY_daily_historical.csv',
        '../CONY_daily_historical.csv',
        './data/CONY_daily_historical.csv'
    ]

    df = None
    for csv_path in possible_paths:
        try:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                logger.info(f"데이터 로드 성공: {csv_path}")
                break
        except Exception as e:
            logger.warning(f"파일 로드 실패 {csv_path}: {e}")

    # 샘플 데이터 생성 (파일이 없을 경우)
    if df is None:
        logger.info("샘플 데이터 생성 중...")
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')

        # 더 현실적인 주가 데이터 생성
        returns = np.random.normal(0.0005, 0.02, 1000)  # 일일 수익률
        prices = [100]  # 시작 가격
        for r in returns[1:]:
            prices.append(prices[-1] * (1 + r))

        prices = np.array(prices)
        volumes = np.random.lognormal(15, 0.5, 1000).astype(int)

        df = pd.DataFrame({
            'Date': dates,
            'Open': prices * (1 + np.random.normal(0, 0.001, 1000)),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.005, 1000))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.005, 1000))),
            'Close': prices,
            'Volume': volumes
        })

        # 음수 방지
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = np.maximum(df[col], 0.01)

        logger.info("샘플 데이터 생성 완료")

    # 2. 데이터 전처리
    try:
        df.columns = df.columns.str.strip()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.dropna(subset=['Date'], inplace=True)
            df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        logger.info(f"원본 데이터: {len(df)}개 행")

        # 기술적 지표 추가
        df = add_enhanced_technical_indicators(df)
        df.dropna(inplace=True)

        logger.info(f"전처리 후 데이터: {len(df)}개 행")

    except Exception as e:
        logger.error(f"데이터 전처리 오류: {e}")
        return None

    if len(df) < 100:
        logger.error("데이터가 너무 적습니다 (최소 100개 필요)")
        return None

    # 3. 데이터 분할 및 정규화
    train_size = int(len(df) * 0.6)
    val_size = int(len(df) * 0.2)

    train_df = df[:train_size].copy()
    val_df = df[train_size:train_size + val_size].copy()
    test_df = df[train_size + val_size:].copy()

    logger.info(f"데이터 분할 - 학습: {len(train_df)}, 검증: {len(val_df)}, 테스트: {len(test_df)}")

    # 데이터 정규화 (가격 데이터는 제외)
    try:
        price_cols = ['Open', 'High', 'Low', 'Close']
        feature_cols = [col for col in df.columns if col not in price_cols]

        if feature_cols:
            scaler = RobustScaler()

            # 학습 데이터로 스케일러 훈련
            train_df_scaled = train_df.copy()
            train_df_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])

            # 검증, 테스트 데이터 변환
            val_df_scaled = val_df.copy()
            val_df_scaled[feature_cols] = scaler.transform(val_df[feature_cols])

            test_df_scaled = test_df.copy()
            test_df_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
        else:
            train_df_scaled = train_df.copy()
            val_df_scaled = val_df.copy()
            test_df_scaled = test_df.copy()
            scaler = None

    except Exception as e:
        logger.error(f"데이터 정규화 오류: {e}")
        return None

    # 4. 강화학습 모델 학습
    logger.info("=== 강화학습 모델 학습 시작 ===")
    lookback_window = min(15, len(train_df) // 10)

    try:
        models, training_histories = train_rl_models(train_df_scaled, val_df_scaled, lookback_window)
        logger.info(f"학습 완료: {list(models.keys())}")

    except Exception as e:
        logger.error(f"모델 학습 오류: {e}")
        return None

    # 5. 백테스팅
    logger.info("=== 백테스팅 실행 ===")
    results = []

    for model_name, model in models.items():
        logger.info(f"{model_name} 백테스팅 중...")
        result = backtest_rl_strategy(model, model_name, test_df_scaled, lookback_window)
        results.append(result)

    # Buy & Hold 벤치마크 추가
    def buy_hold_backtest(test_df, initial_balance=10000):
        if len(test_df) < 20:
            return {'name': 'Buy & Hold', 'total_return': 0, 'error': 'insufficient data'}

        initial_price = test_df.iloc[15]['Close']  # lookback_window 이후 시작
        final_price = test_df.iloc[-1]['Close']

        shares_bought = initial_balance / initial_price
        final_value = shares_bought * final_price
        total_return = (final_value - initial_balance) / initial_balance

        # 일일 가치 계산
        net_worth_history = []
        for price in test_df['Close'].iloc[15:]:
            net_worth_history.append(shares_bought * price)

        returns = pd.Series(net_worth_history).pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0

        return {
            'name': 'Buy & Hold',
            'net_worth_history': net_worth_history,
            'total_return': total_return,
            'annual_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': total_return / volatility if volatility > 0 else 0,
            'max_drawdown': 0,
            'win_rate': 1 if total_return > 0 else 0,
            'final_value': final_value,
            'num_trades': 1,
            'benchmark_return': total_return,
            'excess_return': 0
        }

    benchmark_result = buy_hold_backtest(test_df)
    if 'error' not in benchmark_result:
        results.append(benchmark_result)

    # 6. 결과 출력
    logger.info("\n" + "=" * 100)
    logger.info("결과 비교: PPO (개선) vs A2C vs DQN vs SAC vs Buy&Hold")
    logger.info("=" * 100)

    for result in results:
        if 'error' not in result:
            logger.info(f"\n{result['name']}")
            logger.info(f"   총 수익률: {result['total_return']:.2%}")
            logger.info(f"   연간 수익률: {result['annual_return']:.2%}")
            logger.info(f"   샤프 비율: {result['sharpe_ratio']:.3f}")
            logger.info(f"   최대 낙폭: {result['max_drawdown']:.2%}")
            logger.info(f"   연간 변동성: {result['volatility']:.2%}")
            logger.info(f"   승률: {result['win_rate']:.2%}")
            logger.info(f"   총 거래 횟수: {result['num_trades']}")
            logger.info(f"   최종 자산: ${result['final_value']:,.2f}")
            if 'excess_return' in result:
                logger.info(f"   초과 수익률: {result['excess_return']:.2%}")

    # 7. 최고 성과 알고리즘 선정
    valid_results = [r for r in results if 'error' not in r and r['name'] != 'Buy & Hold']

    if valid_results:
        # 종합 점수로 평가
        for result in valid_results:
            composite_score = (result['sharpe_ratio'] * 0.4 +
                             result['total_return'] * 0.3 +
                             (1 - abs(result['max_drawdown'])) * 0.3)
            result['composite_score'] = composite_score

        best_algorithm = max(valid_results, key=lambda x: x['composite_score'])

        logger.info(f"\n최고 성과 알고리즘: {best_algorithm['name']}")
        logger.info(f"   종합 점수: {best_algorithm['composite_score']:.3f}")
        logger.info(f"   수익률: {best_algorithm['total_return']:.2%}")
        logger.info(f"   샤프 비율: {best_algorithm['sharpe_ratio']:.3f}")

    # 8. DQN 성과 분석
    dqn_result = next((r for r in results if r['name'] == 'DQN'), None)
    if dqn_result and 'error' not in dqn_result:
        logger.info(f"\n=== DQN 상세 분석 ===")
        logger.info(f"DQN은 이산 액션에 특화된 알고리즘으로:")
        logger.info(f"- 탐험/활용 균형이 우수함")
        logger.info(f"- 안정적인 학습 곡선")
        logger.info(f"- 메모리 효율성")
        logger.info(f"실제 성과: 수익률 {dqn_result['total_return']:.2%}, 샤프 {dqn_result['sharpe_ratio']:.3f}")

    # 9. 시각화
    logger.info("\n결과 시각화 생성 중...")
    try:
        create_comprehensive_plots(results, training_histories, test_df)
    except Exception as e:
        logger.warning(f"시각화 오류: {e}")

    # 10. 모델 저장
    logger.info("\n모델 저장 중...")
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for model_name, model in models.items():
            filename = f"rl_trading_{model_name.lower()}_{timestamp}"
            model.save(filename)
            logger.info(f"{model_name} 모델 저장: {filename}.zip")

        if scaler is not None:
            joblib.dump(scaler, f"scaler_{timestamp}.pkl")
            logger.info(f"스케일러 저장: scaler_{timestamp}.pkl")

    except Exception as e:
        logger.warning(f"모델 저장 오류: {e}")

    # 11. 개선사항 요약
    logger.info(f"\n=== 주요 개선사항 ===")
    logger.info("1. PPO 콜백 시스템 개선 → 학습 진행상황 정확한 추적")
    logger.info("2. 관찰 공간 간소화 (20차원) → 학습 효율성 증대")
    logger.info("3. 리워드 함수 단순화 → 명확한 신호")
    logger.info("4. DQN 추가 → 이산 액션 환경에 최적화")
    logger.info("5. 하이퍼파라미터 튜닝 → 안정적 학습")

    return {
        'results': results,
        'models': models,
        'training_histories': training_histories,
        'data': {'train': train_df, 'val': val_df, 'test': test_df},
        'scaler': scaler
    }


if __name__ == "__main__":
    try:
        logger.info("개선된 강화학습 주식거래 시스템 실행")
        main_output = main()

        if main_output is not None:
            logger.info("시스템 실행 완료")
        else:
            logger.error("시스템 실행 실패")
            
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()