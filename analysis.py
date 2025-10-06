# analysis.py - 개선된 주식 거래 환경 및 분석 클래스
# 주요 개선사항:
# 1. 5단계 액션 시스템 (강매도/매도/보유/매수/강매수)
# 2. 개선된 보상 함수 (리스크 조정 수익률, 거래 비용, 드로우다운 페널티 포함)
# 3. 확장된 관찰 공간 (30개 feature, MACD, 볼린저 밴드, ATR 등 추가)
# 4. 현실적인 거래 로직 (비율 기반 거래, 승률 추적)

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


class ImprovedStockTradingEnv(gym.Env):
    """개선된 주식 거래 환경 - 더 현실적인 거래 로직과 보상 시스템"""
    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_balance=10000, lookback_window=20,
                 transaction_cost=0.002, max_position_ratio=1.0):
        super().__init__()
        if len(df) < lookback_window + 10:
            raise ValueError("데이터가 너무 짧습니다.")

        self.df = df.reset_index(drop=False)  # Date 컬럼 보존
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.transaction_cost = transaction_cost
        self.max_position_ratio = max_position_ratio
        self.trade_history = []

        # 액션 공간: 0=강매도, 1=매도, 2=보유, 3=매수, 4=강매수
        self.action_space = spaces.Discrete(5)

        # 관찰 공간 확장 (더 많은 기술적 지표 포함)
        self.observation_space = spaces.Box(
            low=-5, high=5, shape=(30,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.balance = float(self.initial_balance)
        self.shares_held = 0.0
        self.net_worth = float(self.initial_balance)
        self.current_step = self.lookback_window
        self.total_trades = 0
        self.consecutive_holds = 0
        self.last_action = 2  # Start with hold
        self.trade_history = []

        # 성과 추적 변수들
        self.max_net_worth = self.initial_balance
        self.drawdown = 0.0
        self.winning_trades = 0
        self.losing_trades = 0
        return self._get_observation(), self._get_info()

    def _get_observation(self):
        """확장된 관찰 공간 - 30개 feature"""
        safe_step = min(self.current_step, len(self.df) - 1)
        try:
            current_row = self.df.iloc[safe_step]
            current_close = max(0.01, current_row['Close'])

            # 1. 기본 가격 정보 (정규화된 OHLC)
            price_features = np.array([
                (current_row['Open'] / current_close) - 1,
                (current_row['High'] / current_close) - 1,
                (current_row['Low'] / current_close) - 1,
                0.0  # Close는 기준점 (0)
            ])

            # 2. 확장된 기술적 지표 (15개)
            tech_features = np.zeros(15)

            # 이동평균 관련
            if 'SMA_5' in current_row and current_row['SMA_5'] > 0:
                tech_features[0] = np.clip((current_close / current_row['SMA_5']) - 1, -0.5, 0.5)
            if 'SMA_20' in current_row and current_row['SMA_20'] > 0:
                tech_features[1] = np.clip((current_close / current_row['SMA_20']) - 1, -0.5, 0.5)
            if 'EMA_12' in current_row and current_row['EMA_12'] > 0:
                tech_features[2] = np.clip((current_close / current_row['EMA_12']) - 1, -0.5, 0.5)

            # 모멘텀 지표들
            if 'RSI_14' in current_row:
                tech_features[3] = np.clip((current_row['RSI_14'] - 50) / 50, -1, 1)
            if 'Momentum_1' in current_row:
                tech_features[4] = np.clip(current_row['Momentum_1'], -0.2, 0.2)
            if 'Momentum_5' in current_row:
                tech_features[5] = np.clip(current_row['Momentum_5'], -0.5, 0.5)

            # MACD 지표
            if 'MACD' in current_row:
                tech_features[6] = np.clip(current_row['MACD'] / current_close, -0.1, 0.1)
            if 'MACD_Signal' in current_row:
                tech_features[7] = np.clip(current_row['MACD_Signal'] / current_close, -0.1, 0.1)

            # 볼린저 밴드
            if all(col in current_row for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
                bb_range = current_row['BB_Upper'] - current_row['BB_Lower']
                if bb_range > 0:
                    bb_position = (current_close - current_row['BB_Lower']) / bb_range
                    tech_features[8] = np.clip(bb_position, 0, 1)

            # 거래량 관련
            if 'Volume' in current_row:
                volume_ma = self.df['Volume'].rolling(20).mean().iloc[safe_step]
                if volume_ma > 0:
                    tech_features[9] = np.clip(np.log(current_row['Volume'] / volume_ma), -2, 2)

            # 변동성 (ATR)
            if 'ATR' in current_row and current_close > 0:
                tech_features[10] = np.clip(current_row['ATR'] / current_close, 0, 0.1)

            # 가격 모멘텀
            if safe_step >= 5:
                price_momentum = (current_close - self.df['Close'].iloc[safe_step - 5]) / self.df['Close'].iloc[
                    safe_step - 5]
                tech_features[11] = np.clip(price_momentum, -0.2, 0.2)

            # 추가 기술적 지표들 (향후 확장용)
            tech_features[12:15] = 0

            # 3. 포트폴리오 상태 (9개)
            stock_value = self.shares_held * current_close
            total_value = max(0.01, self.balance + stock_value)
            position_ratio = stock_value / total_value

            # 드로우다운 계산
            if total_value > self.max_net_worth:
                self.max_net_worth = total_value
            self.drawdown = (self.max_net_worth - total_value) / self.max_net_worth

            portfolio_features = np.array([
                np.clip(self.balance / self.initial_balance, 0, 3),  # 현금 비율
                np.clip(position_ratio, 0, 1),  # 주식 보유 비율
                np.clip(total_value / self.initial_balance, 0, 5),  # 총 자산 비율
                self.last_action / 4,  # 마지막 액션 정규화
                np.clip(self.consecutive_holds / 20, 0, 1),  # 연속 보유 기간
                np.clip(self.total_trades / 100, 0, 1),  # 총 거래 횟수
                np.clip(self.drawdown, 0, 1),  # 드로우다운
                # 승률 관련
                self.winning_trades / max(1, self.total_trades) if self.total_trades > 0 else 0,
                self.losing_trades / max(1, self.total_trades) if self.total_trades > 0 else 0,
            ])

            # 4. 시장 상황 (2개) - 단기/장기 추세
            market_features = np.zeros(2)
            if safe_step >= self.lookback_window:
                # 단기 추세 (5일)
                short_trend = (current_close - self.df['Close'].iloc[safe_step - 5]) / self.df['Close'].iloc[
                    safe_step - 5]
                # 장기 추세 (20일)
                long_trend = (current_close - self.df['Close'].iloc[safe_step - 20]) / self.df['Close'].iloc[
                    safe_step - 20]
                market_features[0] = np.clip(short_trend, -0.2, 0.2)
                market_features[1] = np.clip(long_trend, -0.5, 0.5)

            observation = np.concatenate([
                price_features,  # 4개
                tech_features,  # 15개
                portfolio_features,  # 9개
                market_features  # 2개
            ])

            # 최종 관측값에서 NaN, inf를 0으로 대체하여 안정성 확보
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)

            return observation.astype(np.float32)

        except Exception as e:
            logger.warning(f"관찰 생성 오류: {e}")
            return np.zeros(30, dtype=np.float32)

    def _get_info(self):
        return {
            'net_worth': self.net_worth,
            'total_trades': self.total_trades,
            'drawdown': self.drawdown,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'balance': self.balance,
            'shares_held': self.shares_held
        }

    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, False, self._get_info()

        prev_net_worth = self.net_worth
        prev_balance = self.balance
        prev_shares = self.shares_held

        # 액션 실행
        self._execute_action(action.item())
        self.current_step += 1

        # 순자산 업데이트
        current_price = self.df.iloc[min(self.current_step, len(self.df) - 1)]['Close']
        self.net_worth = max(0.01, self.balance + self.shares_held * current_price)

        # 개선된 보상 함수
        portfolio_return = (self.net_worth - prev_net_worth) / prev_net_worth if prev_net_worth > 0 else 0

        # 추가 보상/페널티 요소들
        trading_penalty = -0.001 if abs(self.shares_held - prev_shares) > 0 else 0  # 거래 비용
        hold_bonus = 0.0001 if action == 2 and self.shares_held > 0 else 0  # 보유 보너스
        risk_penalty = -max(0, self.drawdown - 0.1) * 0.01  # 큰 드로우다운 페널티

        # 샤프 비율 고려 보상 (변동성 대비 수익률)
        if self.current_step >= 20:
            recent_returns = pd.Series([self.net_worth]).pct_change().dropna()
            if len(recent_returns) > 0 and recent_returns.std() > 0:
                sharpe_bonus = (recent_returns.mean() / recent_returns.std()) * 0.001
            else:
                sharpe_bonus = 0
        else:
            sharpe_bonus = 0

        reward = portfolio_return + trading_penalty + hold_bonus + risk_penalty + sharpe_bonus

        # 종료 조건 (70% 손실 시 종료)
        done = (self.current_step >= len(self.df) - 1 or
                self.net_worth < self.initial_balance * 0.3)

        return self._get_observation(), reward, done, False, self._get_info()

    def _execute_action(self, action):
        """개선된 액션 실행 로직 - 5단계 액션"""
        current_price = self.df.iloc[min(self.current_step, len(self.df) - 1)]['Close']

        if 'Date' in self.df.columns:
            current_date = self.df.iloc[min(self.current_step, len(self.df) - 1)]['Date']
        else:
            current_date = pd.Timestamp.now()

        if current_price <= 0:
            return

        self.last_action = action

        # 액션별 거래량 결정 (0=강매도, 1=매도, 2=보유, 3=매수, 4=강매수)
        if action == 0:  # 강매도 - 전량 매도
            if self.shares_held > 0.01:
                self._sell_shares(1.0, current_price, current_date)

        elif action == 1:  # 매도 - 50% 매도
            if self.shares_held > 0.01:
                self._sell_shares(0.5, current_price, current_date)

        elif action == 2:  # 보유
            self.consecutive_holds += 1

        elif action == 3:  # 매수 - 현금의 30% 투자
            if self.balance > 100:  # 최소 거래 금액
                self._buy_shares(0.3, current_price, current_date)

        elif action == 4:  # 강매수 - 현금의 70% 투자
            if self.balance > 100:
                self._buy_shares(0.7, current_price, current_date)

    def _buy_shares(self, ratio, price, date):
        """매수 실행 함수"""
        self.consecutive_holds = 0
        investment = self.balance * ratio
        cost_with_fee = investment * (1 + self.transaction_cost)

        if cost_with_fee <= self.balance:
            shares_to_buy = investment / price
            self.balance -= cost_with_fee
            self.shares_held += shares_to_buy
            self.total_trades += 1
            self.last_trade_price = price

            self.trade_history.append({
                'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                'type': 'buy',
                'price': price,
                'shares': shares_to_buy,
                'amount': investment
            })

    def _sell_shares(self, ratio, price, date):
        """매도 실행 함수"""
        self.consecutive_holds = 0
        shares_to_sell = self.shares_held * ratio

        if shares_to_sell > 0:
            sale_value = shares_to_sell * price * (1 - self.transaction_cost)
            self.balance += sale_value
            self.shares_held -= shares_to_sell
            self.total_trades += 1

            # 수익/손실 추적
            if self.last_trade_price > 0:
                if price > self.last_trade_price:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1

            self.trade_history.append({
                'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                'type': 'sell',
                'price': price,
                'shares': shares_to_sell,
                'amount': sale_value
            })


class StockAnalyzer:
    """개선된 주식 분석 클래스"""

    def __init__(self):
        self.models_dir = "D:\\StockModelFolder"
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
                logger.info(f"개선된 PPO 모델 로드 완료: {model_name}")
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
        """개선된 주식 분석을 실행합니다."""
        try:
            df_original, _ = download_stock_data(symbol, period, kwargs.get('start_date'), kwargs.get('end_date'))
            if df_original is None:
                return {'error': '데이터를 다운로드할 수 없습니다'}

            df_processed = add_technical_indicators(df_original.copy())
            df_processed.dropna(inplace=True)

            if len(df_processed) < 50:
                return {'error': '분석할 데이터가 부족합니다'}

            # 스케일링 적용
            price_cols = ['Open', 'High', 'Low', 'Close']
            feature_cols = [col for col in df_processed.columns
                            if col not in price_cols and col not in ['Date', 'Target']]

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
        """개선된 백테스팅을 실행합니다."""
        try:
            lookback_window = 20
            env = ImprovedStockTradingEnv(
                df_scaled,
                initial_balance=initial_balance,
                lookback_window=lookback_window
            )

            obs, _ = env.reset()
            done = False

            portfolio_history = []
            actions_history = []

            # 초기 안정화 기간 (5일간 보유)
            burn_in_period = 5
            for _ in range(burn_in_period):
                if not done:
                    portfolio_history.append(env.net_worth)
                    actions_history.append(2)  # 보유
                    obs, reward, done, _, info = env.step(np.array(2))

            # 실제 거래 시작
            while not done:
                portfolio_history.append(env.net_worth)

                # 확률적 액션 선택 (더 현실적인 거래를 위해)
                action, _ = self.model.predict(obs, deterministic=False)
                actions_history.append(action.item())

                obs, reward, done, _, info = env.step(action)

            portfolio_history.append(env.net_worth)

            # 성과 분석
            final_balance = env.net_worth
            total_return = (final_balance - initial_balance) / initial_balance

            # 추가 성과 지표들
            win_rate = env.winning_trades / max(1, env.total_trades) if env.total_trades > 0 else 0
            max_drawdown = env.drawdown

            # 샤프 비율 계산 (간단 버전)
            returns = pd.Series(portfolio_history).pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0

            # --- 오늘의 추천 (신규 투자자 관점) 로직 ---
            # 마지막 관측 상태를 복사하여 포트폴리오 부분만 수정
            recommendation_obs = obs.copy()
            
            # 포트폴리오 상태를 초기 상태(현금 100%, 주식 0%)로 가정
            # 관측 공간에서 포트폴리오 피쳐의 인덱스: 19부터 9개
            # [현금비율, 주식보유비율, 총자산비율, 마지막액션, 연속보유, 총거래, 드로우다운, 승률, 패률]
            new_investor_portfolio_features = np.array([
                1.0,  # 현금 100%
                0.0,  # 주식 0%
                1.0,  # 총 자산은 초기 자본과 동일
                0.5,  # 마지막 액션: 중립(보유)
                0.0,  # 연속 보유 기간 0
                0.0,  # 총 거래 0
                0.0,  # 드로우다운 0
                0.0,  # 승률 0
                0.0   # 패률 0
            ])
            recommendation_obs[19:28] = new_investor_portfolio_features

            # 신규 투자자 관점에서의 행동 예측
            todays_action_code, _ = self.model.predict(recommendation_obs, deterministic=True)
            
            # 신규 투자자에게 더 친화적인 추천으로 변환
            action_map = {0: '관망', 1: '관망', 2: '관망', 3: '매수', 4: '강력매수'}
            todays_action = action_map.get(todays_action_code.item(), '알 수 없음')

            # 차트 데이터 준비
            trade_start_idx = df_scaled.index[0]
            chart_df = df_original[df_original.index >= trade_start_idx].reset_index(drop=True)

            # 포트폴리오 히스토리 길이 조정
            if len(portfolio_history) < len(chart_df):
                last_portfolio_value = portfolio_history[-1] if portfolio_history else initial_balance
                padding = [last_portfolio_value] * (len(chart_df) - len(portfolio_history))
                portfolio_history.extend(padding)
            elif len(portfolio_history) > len(chart_df):
                portfolio_history = portfolio_history[:len(chart_df)]

            # 벤치마크 계산 (매수 후 보유 전략)
            initial_price = chart_df['Close'].iloc[0]
            benchmark_values = (chart_df['Close'] / initial_price) * initial_balance
            benchmark_return = (chart_df['Close'].iloc[-1] - initial_price) / initial_price

            result = {
                'symbol': symbol,
                'period': period,
                'start_date': chart_df['Date'].iloc[0].strftime('%Y-%m-%d'),
                'end_date': chart_df['Date'].iloc[-1].strftime('%Y-%m-%d'),
                'initial_balance': initial_balance,
                'final_balance': final_balance,
                'total_return': total_return,
                'benchmark_return': benchmark_return,
                'total_trades': env.total_trades,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'todays_action': todays_action,
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