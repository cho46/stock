import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import joblib
import os
import logging
import traceback
from typing import Dict

from stable_baselines3 import PPO
from sklearn.preprocessing import RobustScaler

from utils import download_stock_data, add_technical_indicators

logger = logging.getLogger(__name__)

# 훈련과 분석에 공통으로 사용될 환경
class OptimizedStockTradingEnv(gym.Env):
    """최적화된 주식 거래 환경"""
    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_balance=10000, lookback_window=15,
                 max_position_ratio=0.3, transaction_cost=0.002):
        super().__init__()
        if len(df) < lookback_window + 10: raise ValueError("데이터가 너무 짧습니다.")
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.max_position_ratio = max_position_ratio
        self.transaction_cost = transaction_cost
        self.trade_history = []
        
        self.action_space = spaces.Discrete(3) # 0: Sell, 1: Hold, 2: Buy
        self.observation_space = spaces.Box(low=-5, high=5, shape=(20,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.balance = float(self.initial_balance)
        self.shares_held = 0.0
        self.net_worth = float(self.initial_balance)
        self.current_step = self.lookback_window
        self.total_trades = 0
        self.consecutive_holds = 0
        self.last_action = 1 # Start with hold
        self.trade_history = [] # Clear trade history on reset
        return self._get_observation(), self._get_info()

    def _get_observation(self):
        safe_step = min(self.current_step, len(self.df) - 1)
        try:
            current_row = self.df.iloc[safe_step]
            current_close = max(0.01, current_row['Close'])
            
            price_features = np.array([
                current_row['Open'] / current_close, current_row['High'] / current_close,
                current_row['Low'] / current_close, 1.0
            ])

            tech_features = np.zeros(10)
            if 'SMA_5' in current_row and current_row['SMA_5'] > 0: tech_features[0] = np.clip(current_close / current_row['SMA_5'] - 1, -0.5, 0.5)
            if 'SMA_20' in current_row and current_row['SMA_20'] > 0: tech_features[1] = np.clip(current_close / current_row['SMA_20'] - 1, -0.5, 0.5)
            if 'RSI_14' in current_row: tech_features[2] = np.clip((current_row['RSI_14'] - 50) / 50, -1, 1)
            if 'Momentum_1' in current_row: tech_features[3] = np.clip(current_row['Momentum_1'], -0.2, 0.2)
            if 'Momentum_5' in current_row: tech_features[4] = np.clip(current_row['Momentum_5'], -0.5, 0.5)

            stock_value = self.shares_held * current_close
            total_value = max(0.01, self.balance + stock_value)
            position_ratio = stock_value / total_value

            portfolio_features = np.array([
                np.clip(self.balance / self.initial_balance, 0, 2),
                np.clip(position_ratio, 0, 1),
                np.clip(total_value / self.initial_balance, 0, 3),
                self.last_action / 2,
                np.clip(self.consecutive_holds / 10, 0, 1),
                np.clip(self.total_trades / 50, 0, 1)
            ])

            observation = np.concatenate([price_features, tech_features, portfolio_features])
            return observation.astype(np.float32)
        except Exception as e:
            logger.warning(f"관찰 생성 오류: {e}")
            return np.zeros(20, dtype=np.float32)

    def _get_info(self):
        return {'net_worth': self.net_worth, 'total_trades': self.total_trades}

    def step(self, action):
        if self.current_step >= len(self.df) - 1: return self._get_observation(), 0, True, False, self._get_info()
        
        prev_net_worth = self.net_worth
        self._execute_action(action.item()) # Use action.item() to get scalar value
        self.current_step += 1
        
        current_price = self.df.iloc[min(self.current_step, len(self.df) - 1)]['Close']
        self.net_worth = max(0.01, self.balance + self.shares_held * current_price)
        
        reward = (self.net_worth - prev_net_worth) / prev_net_worth if prev_net_worth > 0 else 0
        done = self.current_step >= len(self.df) - 1 or self.net_worth < self.initial_balance * 0.2
        
        return self._get_observation(), reward, done, False, self._get_info()

    def _execute_action(self, action):
        current_price = self.df.iloc[min(self.current_step, len(self.df) - 1)]['Close']
        current_date = self.df.iloc[min(self.current_step, len(self.df) - 1)]['Date']
        if current_price <= 0: return

        self.last_action = action
        if action == 2:  # Buy
            self.consecutive_holds = 0
            cost = self.balance * self.max_position_ratio
            if cost > 10:
                shares_to_buy = cost / (current_price * (1 + self.transaction_cost))
                self.balance -= cost
                self.shares_held += shares_to_buy
                self.total_trades += 1
                self.trade_history.append({'date': current_date.strftime('%Y-%m-%d'), 'type': 'buy', 'price': current_price})
        elif action == 0:  # Sell
            self.consecutive_holds = 0
            if self.shares_held > 0.01:
                shares_to_sell = self.shares_held * 0.5 # Sell 50% of holdings
                sale_value = shares_to_sell * current_price * (1 - self.transaction_cost)
                self.balance += sale_value
                self.shares_held -= shares_to_sell
                self.total_trades += 1
                self.trade_history.append({'date': current_date.strftime('%Y-%m-%d'), 'type': 'sell', 'price': current_price})
        else: # Hold
            self.consecutive_holds += 1

class StockAnalyzer:
    """웹 애플리케이션용 주식 분석 클래스"""
    def __init__(self):
        self.models_dir = "models"
        self.model = None
        self.scaler = None

    def load_model(self, model_name: str, user_id: str):
        """선택된 모델과 스케일러를 로드합니다."""
        try:
            user_models_path = os.path.join(self.models_dir, user_id)
            model_path = os.path.join(user_models_path, model_name)
            scaler_name = model_name.replace('.zip', '_scaler.pkl')
            scaler_path = os.path.join(user_models_path, scaler_name)

            if os.path.exists(model_path):
                self.model = PPO.load(model_path)
                logger.info(f"PPO 모델 로드 완료: {model_name}")
            else:
                logger.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
                return False

            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info(f"스케일러 로드 완료: {scaler_name}")
            else:
                self.scaler = None
                logger.warning(f"스케일러 파일을 찾을 수 없습니다: {scaler_path}")
            
            return True
        except Exception as e:
            logger.error(f"모델 로드 중 오류: {e}")
            return False

    def analyze_stock(self, symbol: str, period: str, initial_balance: float, **kwargs) -> Dict:
        """주식 분석을 실행합니다."""
        try:
            df_original = download_stock_data(symbol, period, kwargs.get('start_date'), kwargs.get('end_date'))
            if df_original is None: return {'error': '데이터를 다운로드할 수 없습니다'}

            df_processed = add_technical_indicators(df_original.copy())
            df_processed.dropna(inplace=True)
            if len(df_processed) < 50: return {'error': '분석할 데이터가 부족합니다'}

            # 스케일링 로직 수정
            price_cols = ['Open', 'High', 'Low', 'Close']
            feature_cols = [col for col in df_processed.columns if col not in price_cols and col not in ['Date', 'Target']]
            
            df_scaled = df_processed.copy()
            if self.scaler and all(col in df_scaled.columns for col in feature_cols):
                try:
                    df_scaled[feature_cols] = self.scaler.transform(df_scaled[feature_cols])
                    logger.info("데이터 스케일링 적용 완료.")
                except Exception as e:
                    logger.warning(f"스케일링 실패, 원본 데이터 사용: {e}")
            
            return self._run_backtest(df_scaled, df_original, initial_balance, symbol, period, **kwargs)
        except Exception as e:
            logger.error(f"분석 중 오류: {e}")
            return {'error': f'분석 중 오류가 발생했습니다: {str(e)}'}

    def _run_backtest(self, df_scaled: pd.DataFrame, df_original: pd.DataFrame,
                     initial_balance: float, symbol: str, period: str, **kwargs) -> Dict:
        """백테스팅을 실행하고 Plotly.js에 맞는 형식의 데이터를 반환합니다."""
        try:
            lookback_window = 15
            env = OptimizedStockTradingEnv(df_scaled, initial_balance=initial_balance, lookback_window=lookback_window)
            
            obs, _ = env.reset()
            done = False
            
            portfolio_history = []
            while not done:
                portfolio_history.append(env.net_worth)
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
            portfolio_history.append(env.net_worth)

            final_balance = env.net_worth
            total_return = (final_balance - initial_balance) / initial_balance

            trade_start_idx = df_scaled.index[0]
            chart_df = df_original[df_original.index >= trade_start_idx].iloc[:len(portfolio_history)]

            initial_price = chart_df['Close'].iloc[0]
            benchmark_values = (chart_df['Close'] / initial_price) * initial_balance
            benchmark_return = (chart_df['Close'].iloc[-1] - initial_price) / initial_price

            result = {
                'symbol': symbol, 'period': period,
                'start_date': chart_df['Date'].iloc[0].strftime('%Y-%m-%d'),
                'end_date': chart_df['Date'].iloc[-1].strftime('%Y-%m-%d'),
                'initial_balance': initial_balance, 'final_balance': final_balance,
                'total_return': total_return, 'benchmark_return': benchmark_return,
                'total_trades': env.total_trades,
                'chart_data': {
                    'dates': chart_df['Date'].dt.strftime('%Y-%m-%d').tolist(),
                    'open': chart_df['Open'].tolist(),
                    'high': chart_df['High'].tolist(),
                    'low': chart_df['Low'].tolist(),
                    'close': chart_df['Close'].tolist(),
                    'portfolio': portfolio_history,
                    'benchmark': benchmark_values.tolist(),
                    'trades': env.trade_history
                }
            }
            return result
        except Exception as e:
            logger.error(f"백테스팅 오류: {e}")
            traceback.print_exc()
            return {'error': f'백테스팅 중 오류: {str(e)}'}