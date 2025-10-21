# main_app.py - 개선된 Flask 메인 애플리케이션
# 주요 개선사항:
# 1. 개선된 환경 클래스 사용 (ImprovedStockTradingEnv)
# 2. 고급 기술적 지표 지원
# 3. 에러 핸들링 강화
# 4. 성과 지표 확장 (승률, 샤프 비율, 최대 낙폭)
# 5. 사용자 경험 향상 (더 나은 피드백, 로딩 상태)

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response, stream_with_context
from training import run_training_process
from analysis import StockAnalyzer
import os
import glob
import json
import logging
import click
from datetime import datetime
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import or_, text
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
from utils import download_from_blob_storage, delete_blob

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask 앱 및 기본 설정
app = Flask(__name__)

# --- 보안 및 설정 ---
# 시크릿 키와 데이터베이스 URI를 환경 변수에서 불러옵니다.
# 로컬 개발 시에는 .env 파일을 통해 이 값들을 설정합니다.
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')

# 이 값들이 설정되지 않았다면, 앱 실행을 중단시켜 설정을 강제합니다.
if not app.config['SECRET_KEY'] or not app.config['SQLALCHEMY_DATABASE_URI']:
    raise ValueError("SECRET_KEY와 DATABASE_URL 환경 변수를 설정해야 합니다.")

logger.info("환경 변수로부터 데이터베이스 설정을 로드했습니다.")


app.config['JSON_AS_ASCII'] = False
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_recycle': 3600,
    'pool_pre_ping': True
}

#Virtual Machines, BS Series, B1s
db = SQLAlchemy(app)

# --- DB 연결 테스트 코드 ---
try:
    with app.app_context():
        # db.engine.connect()를 사용하여 실제 연결을 시도하고, 간단한 쿼리를 실행합니다.
        connection = db.engine.connect()
        connection.execute(text("SELECT 1"))
        connection.close()
        logger.info("✅ 데이터베이스 연결 성공: 애플리케이션이 성공적으로 데이터베이스에 연결되었습니다.")
except Exception as e:
    logger.error("❌ 데이터베이스 연결 실패: 시작 시점에 데이터베이스에 연결할 수 없습니다. DATABASE_URL 또는 방화벽 설정을 확인하세요.")
    logger.error(f"상세 오류: {e}")
# --- DB 연결 테스트 코드 끝 ---

# Flask-Login 설정
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = '/login'
login_manager.login_message = '로그인이 필요합니다.'
login_manager.login_message_category = 'warning'


# 사용자 모델 정의 (기존 DB 스키마에 맞게 수정)
class UsersInfo(UserMixin, db.Model):
    __tablename__ = 'users_info'
    users_seq = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String(50), unique=True, nullable=False, index=True)
    password = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(50), nullable=False)
    hp = db.Column(db.String(13), nullable=False)
    age = db.Column(db.Integer, nullable=False, default=0)
    email = db.Column(db.String(100), nullable=False)
    isadult = db.Column(db.String(5), nullable=False, default='False')

    # created_at과 last_login은 기존 테이블에 없다면 주석 처리
    # created_at = db.Column(db.DateTime, default=datetime.utcnow)
    # last_login = db.Column(db.DateTime)

    def get_id(self):
        return self.user_id

    def update_last_login(self):
        """last_login 컬럼이 있는 경우에만 업데이트"""
        try:
            if hasattr(self, 'last_login'):
                self.last_login = datetime.utcnow()
                db.session.commit()
        except Exception as e:
            logger.warning(f"로그인 시간 업데이트 실패: {e}")


# 신규: 모델 파일 정보 저장을 위한 테이블
class ModelFiles(db.Model):
    __tablename__ = 'model_files'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String(50), nullable=False)
    model_name = db.Column(db.String(100), nullable=False, index=True)
    symbol = db.Column(db.String(20))
    strategy = db.Column(db.String(20))
    training_date = db.Column(db.DateTime, default=datetime.utcnow)
    blob_model_path = db.Column(db.String(255))
    blob_scaler_path = db.Column(db.String(255))
    blob_metadata_path = db.Column(db.String(255))
    is_deleted = db.Column(db.Boolean, default=False)

    __table_args__ = (db.UniqueConstraint('user_id', 'model_name', name='_user_model_name_uc'),)


# 미국 주식 목록 모델
class UsStockInfo(db.Model):
    __tablename__ = 'us_stock_info'
    ticker = db.Column(db.String(20), primary_key=True, nullable=False)
    name = db.Column(db.String(255), nullable=False)
    exchange = db.Column(db.String(20), nullable=False)

    def __repr__(self):
        return f'<UsStockInfo {self.ticker} - {self.name}>'


# 사용자 로그 모델
class UserLogs(db.Model):
    __tablename__ = 'user_logs'
    visit_seq = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String(50), nullable=False)
    address = db.Column(db.Text, nullable=False)
    time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_agent = db.Column(db.Text)
    ip_address = db.Column(db.String(45))


# 모델 성과 추적 테이블
class ModelPerformance(db.Model):
    __tablename__ = 'model_performance'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String(50), nullable=False)
    model_name = db.Column(db.String(100), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    strategy = db.Column(db.String(20), default='balanced')
    total_return = db.Column(db.Float)
    benchmark_return = db.Column(db.Float)
    win_rate = db.Column(db.Float)
    max_drawdown = db.Column(db.Float)
    sharpe_ratio = db.Column(db.Float)
    total_trades = db.Column(db.Integer)
    analysis_date = db.Column(db.DateTime, default=datetime.utcnow)


# 사용자 즐겨찾기 종목 모델
class UserFavorites(db.Model):
    __tablename__ = 'user_favorites'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String(50), db.ForeignKey('users_info.user_id'), nullable=False)
    ticker = db.Column(db.String(20), nullable=False)
    __table_args__ = (db.UniqueConstraint('user_id', 'ticker', name='_user_ticker_uc'),)


# 실시간 가격 캐시 모델
class RealtimeStockPrice(db.Model):
    __tablename__ = 'realtime_stock_prices'
    ticker = db.Column(db.String(20), primary_key=True)
    price = db.Column(db.Float)
    change = db.Column(db.Float)
    percent_change = db.Column(db.Float)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# 포트폴리오 보유 종목 모델
class PortfolioHolding(db.Model):
    __tablename__ = 'portfolio_holdings'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String(50), db.ForeignKey('users_info.user_id'), nullable=False)
    ticker = db.Column(db.String(20), nullable=False)
    quantity = db.Column(db.Float, nullable=False)
    average_cost = db.Column(db.Float, nullable=False)
    __table_args__ = (db.UniqueConstraint('user_id', 'ticker', name='_user_holding_uc'),)


# 포트폴리오 거래 내역 모델
class PortfolioTransaction(db.Model):
    __tablename__ = 'portfolio_transactions'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String(50), db.ForeignKey('users_info.user_id'), nullable=False)
    ticker = db.Column(db.String(20), nullable=False)
    transaction_type = db.Column(db.String(4), nullable=False)  # 'BUY' or 'SELL'
    quantity = db.Column(db.Float, nullable=False)
    price = db.Column(db.Float, nullable=False)
    transaction_date = db.Column(db.DateTime, default=datetime.utcnow)



# --- 헬퍼 함수들 ---

def get_prices_from_cache_or_fetch(tickers: list):
    """DB 캐시에서 가격을 조회하고, 오래된 데이터는 API로 새로 가져오되, 실패 시 캐시된 데이터를 우선 사용합니다."""
    import yfinance as yf
    import pandas as pd
    from datetime import datetime, timedelta

    if not tickers:
        return {}

    final_prices = {}
    stale_tickers = []

    # 1. 요청된 모든 종목에 대해 캐시된 데이터를 먼저 모두 불러와 결과에 저장
    cached_prices = RealtimeStockPrice.query.filter(RealtimeStockPrice.ticker.in_(tickers)).all()
    cache_fresh_time = datetime.utcnow() - timedelta(seconds=60)

    for price_obj in cached_prices:
        final_prices[price_obj.ticker] = {
            'price': price_obj.price,
            'change': price_obj.change,
            'percent_change': price_obj.percent_change
        }
        # 데이터가 오래되었는지 확인
        if price_obj.last_updated < cache_fresh_time:
            stale_tickers.append(price_obj.ticker)

    # 2. 캐시에 아예 존재하지 않는 종목 식별
    cached_tickers_set = {p.ticker for p in cached_prices}
    missing_tickers = [t for t in tickers if t not in cached_tickers_set]

    # 3. 오래된 종목과 없는 종목을 합쳐 새로 조회할 목록 생성
    to_fetch = stale_tickers + missing_tickers

    if to_fetch:
        try:
            yf_data = yf.download(to_fetch, period='2d', progress=False, timeout=10)
            if not yf_data.empty:
                for ticker_symbol in to_fetch:
                    try:
                        hist = yf_data['Close'] if len(to_fetch) == 1 else yf_data['Close'][ticker_symbol]
                        if len(hist) >= 2 and not hist.isnull().all():
                            last_price = hist.iloc[-1]
                            prev_close = hist.iloc[-2]

                            if pd.isna(last_price) or pd.isna(prev_close):
                                continue

                            change = last_price - prev_close
                            percent_change = (change / prev_close) * 100 if prev_close != 0 else 0

                            # 성공 시, 결과와 DB 모두 업데이트
                            final_prices[ticker_symbol] = {
                                'price': last_price,
                                'change': change,
                                'percent_change': percent_change
                            }
                            
                            price_entry = RealtimeStockPrice.query.get(ticker_symbol)
                            if price_entry:
                                price_entry.price = last_price
                                price_entry.change = change
                                price_entry.percent_change = percent_change
                                price_entry.last_updated = datetime.utcnow()
                            else:
                                price_entry = RealtimeStockPrice(
                                    ticker=ticker_symbol, price=last_price, 
                                    change=change, percent_change=percent_change
                                )
                                db.session.add(price_entry)
                    except Exception as e:
                        logger.warning(f'Helper: {ticker_symbol} 데이터 처리 실패: {e}')
                        # 실패 시, 기존의 stale 데이터가 final_prices에 남아있게 됨

                db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.error(f"Helper: yfinance 데이터 조회 실패: {e}")

    return final_prices


# 사용자 로더 함수
@login_manager.user_loader
def load_user(user_id):
    return UsersInfo.query.filter_by(user_id=user_id).first()


# 전역 에러 핸들러
@app.errorhandler(404)
def not_found(error):
    """404 에러 처리 - 템플릿이 없을 경우 대비"""
    return jsonify({
        "error": "페이지를 찾을 수 없습니다.",
        "status": 404
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """500 에러 처리"""
    db.session.rollback()
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "error": "서버 내부 오류가 발생했습니다.",
        "status": 500
    }), 500


# --- API 라우트들 ---

@app.route('/train', methods=['POST'])
@login_required
def train_model_route():
    """개선된 모델 훈련 API - DB에 모델 정보 저장 로직 추가"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "요청 데이터가 없습니다."}), 400

        required_fields = ['symbol', 'period', 'model_name']
        missing_fields = [field for field in required_fields if not data.get(field)]
        if missing_fields:
            return jsonify({
                "status": "error",
                "message": f"필수 필드가 누락되었습니다: {', '.join(missing_fields)}"
            }), 400

        symbol = data['symbol'].upper().strip()
        period = data['period']
        model_name = data['model_name'].strip()
        strategy = data.get('strategy', 'balanced')
        user_id = current_user.get_id()

        # 입력값 검증 (기존과 동일)
        # ...

        logger.info(f"훈련 시작 - 사용자: {user_id}, 종목: {symbol}, 전략: {strategy}")

        def training_stream():
            final_model_info = None
            # 훈련 프로세스에서 오는 스트림을 그대로 클라이언트에게 전달
            for response_part in run_training_process(symbol, period, user_id, model_name, strategy):
                try:
                    # 각 JSON 부분을 파싱하여 상태 확인
                    response_json = json.loads(response_part)
                    if response_json.get("status") == "success":
                        final_model_info = response_json.get("model_info")
                except json.JSONDecodeError:
                    # JSON 파싱이 불가능한 경우, 원본 스트림을 그대로 전달
                    pass
                yield response_part
            
            # 스트림이 끝나고, 성공적으로 모델 정보가 반환되었으면 DB에 저장
            if final_model_info:
                try:
                    # 동일한 이름의 모델이 있는지 확인 (삭제된 것도 포함)
                    model_entry = ModelFiles.query.filter_by(
                        user_id=user_id, 
                        model_name=final_model_info['model_name']
                    ).first()

                    if model_entry:
                        # 모델이 존재하면, 정보 업데이트
                        logger.info(f"기존 모델 정보를 업데이트합니다: {model_entry.model_name}")
                        model_entry.symbol = final_model_info['symbol']
                        model_entry.strategy = final_model_info['strategy']
                        model_entry.training_date = datetime.now()
                        model_entry.blob_model_path = final_model_info['blob_model_path']
                        model_entry.blob_scaler_path = final_model_info['blob_scaler_path']
                        model_entry.blob_metadata_path = final_model_info['blob_metadata_path']
                        model_entry.is_deleted = False # 혹시 삭제되었던 모델이면 활성화
                    else:
                        # 모델이 없으면, 새로 생성
                        logger.info(f"새 모델 정보를 데이터베이스에 추가합니다: {final_model_info['model_name']}")
                        model_entry = ModelFiles(
                            user_id=user_id,
                            model_name=final_model_info['model_name'],
                            symbol=final_model_info['symbol'],
                            strategy=final_model_info['strategy'],
                            training_date=datetime.now(),
                            blob_model_path=final_model_info['blob_model_path'],
                            blob_scaler_path=final_model_info['blob_scaler_path'],
                            blob_metadata_path=final_model_info['blob_metadata_path']
                        )
                        db.session.add(model_entry)
                    
                    db.session.commit()
                    logger.info(f"DB 작업 완료: {final_model_info['model_name']}")

                except Exception as e:
                    db.session.rollback()
                    logger.error(f"DB에 모델 정보 저장 실패: {e}")

        return Response(
            stream_with_context(training_stream()),
            content_type='application/json; charset=utf-8'
        )

    except Exception as e:
        logger.error(f"훈련 요청 처리 중 오류: {e}")
        return jsonify({"status": "error", "message": "훈련 요청 처리 중 오류가 발생했습니다."}), 500


@app.route('/analyze', methods=['POST'])
@login_required
def analyze_stock_route():
    """개선된 주식 분석 API - DB 연동"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "요청 데이터가 없습니다."}), 400

        required_fields = ['model_name', 'symbol', 'period']
        missing_fields = [field for field in required_fields if not data.get(field)]
        if missing_fields:
            return jsonify({
                "error": f"필수 필드가 누락되었습니다: {', '.join(missing_fields)}"
            }), 400

        model_name = data['model_name']
        symbol = data['symbol'].upper().strip()
        period = data['period']
        initial_balance = float(data.get('initial_balance', 10000))
        user_id = current_user.get_id()

        logger.info(f"분석 요청 - 사용자: {user_id}, 종목: {symbol}, 모델: {model_name}")

        # DB에서 모델 정보 조회
        model_record = ModelFiles.query.filter_by(user_id=user_id, model_name=model_name, is_deleted=False).first()
        if not model_record:
            return jsonify({"error": "지정된 모델을 찾을 수 없습니다."}), 404

        # 분석기 생성 및 모델 로드
        analyzer = StockAnalyzer()
        if not analyzer.load_model(model_record):
            return jsonify({"error": "모델을 불러오는 데 실패했습니다. Blob Storage 또는 파일을 확인해주세요."}), 500

        # 주식 분석 실행
        result = analyzer.analyze_stock(
            symbol=symbol,
            period=period,
            initial_balance=initial_balance
        )

        if 'error' in result:
            logger.warning(f"분석 실패 - {result['error']}")
            return jsonify(result), 500

        # 성과 기록 저장
        try:
            performance_record = ModelPerformance(
                user_id=user_id,
                model_name=model_name,
                symbol=symbol,
                strategy=model_record.strategy, # DB에서 가져온 전략 정보 사용
                total_return=result.get('total_return'),
                benchmark_return=result.get('benchmark_return'),
                win_rate=result.get('win_rate'),
                max_drawdown=result.get('max_drawdown'),
                sharpe_ratio=result.get('sharpe_ratio'),
                total_trades=result.get('total_trades')
            )
            db.session.add(performance_record)
            db.session.commit()
            logger.info(f"성과 기록 저장 완료 - {symbol}")
        except Exception as e:
            db.session.rollback()
            logger.warning(f"성과 기록 저장 실패: {e}")

        logger.info(f"분석 완료 - 수익률: {result.get('total_return', 0) * 100:.2f}%")
        return jsonify(result)

    except ValueError as e:
        logger.error(f"분석 요청 파라미터 오류: {e}")
        return jsonify({"error": "잘못된 파라미터입니다. 입력값을 확인해주세요."}), 400
    except Exception as e:
        logger.error(f"분석 요청 처리 중 오류: {e}")
        return jsonify({"error": "분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."}), 500


@app.route('/api/search_us_stock')
def search_us_stock_route():
    """미국 주식 검색 API"""
    try:
        query = request.args.get('query', '').strip()
        if not query or len(query) < 2:
            return jsonify([])

        search_term = f"%{query}%"
        results = UsStockInfo.query.filter(
            or_(
                UsStockInfo.ticker.ilike(search_term),
                UsStockInfo.name.ilike(search_term)
            )
        ).limit(10).all()

        return jsonify([{
            'ticker': r.ticker,
            'name': r.name,
            'exchange': r.exchange
        } for r in results])

    except Exception as e:
        logger.error(f"주식 검색 오류: {e}")
        return jsonify([])


@app.route('/model')
@login_required
def get_models():
    """사용자 모델 목록을 데이터베이스에서 조회하는 API"""
    try:
        user_id = current_user.get_id()
        
        # is_deleted가 False인 모델만 조회
        user_models = ModelFiles.query.filter_by(user_id=user_id, is_deleted=False)\
                                      .order_by(ModelFiles.training_date.desc()).all()

        model_files = []
        for model_record in user_models:
            model_files.append({
                'filename': model_record.model_name,
                'symbol': model_record.symbol,
                'strategy': model_record.strategy,
                'created_date': model_record.training_date.isoformat() if model_record.training_date else 'Unknown',
                'expected_return': 0 # 이 값은 필요 시 추후 계산하여 채워넣을 수 있음
            })

        logger.info(f"DB에서 찾은 모델 수: {len(model_files)}")
        return jsonify(model_files)

    except Exception as e:
        logger.error(f"DB에서 모델 목록 조회 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "모델 목록을 불러오는 중 오류가 발생했습니다."}), 500


@app.route('/api/model_performance/<model_name>')
@login_required
def get_model_performance(model_name):
    """모델 성과 이력 조회 API"""
    try:
        user_id = current_user.get_id()
        performances = ModelPerformance.query.filter_by(
            user_id=user_id,
            model_name=model_name
        ).order_by(ModelPerformance.analysis_date.desc()).limit(10).all()

        return jsonify([{'symbol': p.symbol,
            'total_return': p.total_return,
            'benchmark_return': p.benchmark_return,
            'win_rate': p.win_rate,
            'max_drawdown': p.max_drawdown,
            'sharpe_ratio': p.sharpe_ratio,
            'total_trades': p.total_trades,
            'analysis_date': p.analysis_date.isoformat() if p.analysis_date else None
        } for p in performances])

    except Exception as e:
        logger.error(f"모델 성과 조회 오류: {e}")
        return jsonify([])


@app.route('/api/market_overview')
def market_overview():
    """메인 페이지의 시장 현황 데이터 API (캐시 기반)"""
    if not current_user.is_authenticated:
        return jsonify([])

    user_id = current_user.get_id()
    favorites = UserFavorites.query.filter_by(user_id=user_id).all()
    if not favorites:
        return jsonify([])

    tickers = [fav.ticker for fav in favorites]
    price_data = get_prices_from_cache_or_fetch(tickers)

    market_data = []
    for ticker_symbol in tickers:
        stock_info = UsStockInfo.query.get(ticker_symbol)
        name = stock_info.name if stock_info else ticker_symbol
        data = price_data.get(ticker_symbol)

        market_data.append({
            'ticker': ticker_symbol,
            'name': name,
            'price': data.get('price') if data else None,
            'change': data.get('change') if data else None,
            'percent_change': data.get('percent_change') if data else None
        })
    
    return jsonify(market_data)


@app.route('/api/stocks_list')
@login_required
def stocks_list():
    """전체 종목 리스트 API (캐시 기반)"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 30, type=int)
        search_query = request.args.get('search', '', type=str).strip()

        query = UsStockInfo.query
        if search_query:
            search_term = f"%{search_query}%"
            query = query.filter(or_(
                UsStockInfo.ticker.ilike(search_term),
                UsStockInfo.name.ilike(search_term)
            ))
        
        # MSSQL에서 페이지네이션을 사용하기 위해 ORDER BY 절 추가
        paginated_stocks = query.order_by(UsStockInfo.ticker).paginate(page=page, per_page=per_page, error_out=False)
        db_stocks = paginated_stocks.items
        tickers = [stock.ticker for stock in db_stocks]

        user_favorites = UserFavorites.query.filter_by(user_id=current_user.get_id()).all()
        favorite_tickers = {fav.ticker for fav in user_favorites}

        price_data = get_prices_from_cache_or_fetch(tickers)

        results = []
        for stock in db_stocks:
            data = price_data.get(stock.ticker)
            results.append({
                'ticker': stock.ticker,
                'name': stock.name,
                'price': data.get('price') if data else None,
                'change': data.get('change') if data else None,
                'percent_change': data.get('percent_change') if data else None,
                'is_favorite': stock.ticker in favorite_tickers
            })

        return jsonify({
            'stocks': results,
            'pagination': {
                'page': paginated_stocks.page,
                'per_page': paginated_stocks.per_page,
                'total_pages': paginated_stocks.pages,
                'total_items': paginated_stocks.total,
                'has_next': paginated_stocks.has_next,
                'has_prev': paginated_stocks.has_prev
            }
        })

    except Exception as e:
        logger.error(f"종목 리스트 API 오류: {e}")
        return jsonify({"error": "데이터를 불러오는 중 오류가 발생했습니다."}), 500


@app.route('/api/favorites/add', methods=['POST'])
@login_required
def add_favorite():
    """즐겨찾기 추가 API"""
    data = request.get_json()
    ticker = data.get('ticker')
    if not ticker:
        return jsonify({'status': 'error', 'message': 'Ticker is required'}), 400

    try:
        user_id = current_user.get_id()
        existing = UserFavorites.query.filter_by(user_id=user_id, ticker=ticker).first()
        if not existing:
            new_fav = UserFavorites(user_id=user_id, ticker=ticker)
            db.session.add(new_fav)
            db.session.commit()
        return jsonify({'status': 'success', 'message': 'Favorite added'})
    except Exception as e:
        db.session.rollback()
        logger.error(f"즐겨찾기 추가 오류: {e}")
        return jsonify({'status': 'error', 'message': 'Failed to add favorite'}), 500


@app.route('/api/favorites/remove', methods=['POST'])
@login_required
def remove_favorite():
    """즐겨찾기 제거 API"""
    data = request.get_json()
    ticker = data.get('ticker')
    if not ticker:
        return jsonify({'status': 'error', 'message': 'Ticker is required'}), 400

    try:
        user_id = current_user.get_id()
        fav = UserFavorites.query.filter_by(user_id=user_id, ticker=ticker).first()
        if fav:
            db.session.delete(fav)
            db.session.commit()
        return jsonify({'status': 'success', 'message': 'Favorite removed'})
    except Exception as e:
        db.session.rollback()
        logger.error(f"즐겨찾기 제거 오류: {e}")
        return jsonify({'status': 'error', 'message': 'Failed to remove favorite'}), 500


# --- 웹 페이지 라우트들 ---

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')


@app.route('/chart')
def chart_page():
    """차트 페이지"""
    return render_template('chart.html')


@app.route('/analysis')
def analysis_page():
    """분석 페이지"""
    return render_template('analysis.html')


@app.route('/models')
@login_required
def models_page():
    """모델 관리 페이지"""
    return render_template('models.html')


@app.route('/portfolio')
@login_required
def portfolio_page():
    """포트폴리오 페이지"""
    return render_template('portfolio.html')


@app.route('/stocks')
@login_required
def stocks_page():
    """종목보기 페이지"""
    return render_template('stocks.html')


@app.route('/api/models/delete', methods=['POST'])
@login_required
def delete_model():
    """모델 삭제 API - DB 및 Blob Storage 연동"""
    data = request.get_json()
    filename = data.get('filename')
    if not filename:
        return jsonify({'status': 'error', 'message': 'Filename is required'}), 400

    try:
        user_id = current_user.get_id()
        
        # DB에서 모델 정보 조회
        model_to_delete = ModelFiles.query.filter_by(user_id=user_id, model_name=filename, is_deleted=False).first()

        if not model_to_delete:
            return jsonify({'status': 'error', 'message': 'Model not found'}), 404

        # Blob Storage에서 파일들 삭제
        files_to_delete_in_blob = [
            model_to_delete.blob_model_path,
            model_to_delete.blob_scaler_path,
            model_to_delete.blob_metadata_path
        ]
        
        deleted_count = 0
        for blob_path in files_to_delete_in_blob:
            if blob_path and delete_blob(blob_path):
                deleted_count += 1
                logger.info(f"Blob에서 파일 삭제됨: {blob_path}")
            else:
                logger.warning(f"Blob 파일 삭제 실패 또는 경로 없음: {blob_path}")

        # DB에서 soft delete 처리
        model_to_delete.is_deleted = True
        db.session.commit()
        logger.info(f"DB에서 모델 레코드 soft delete 처리됨: {filename}")

        if deleted_count > 0:
            return jsonify({'status': 'success', 'message': f'{filename} and related files marked as deleted.'})
        else:
            # DB에서는 삭제 처리되었지만 Blob 파일 삭제에 실패한 경우
            return jsonify({'status': 'warning', 'message': 'Model record was deleted, but failed to delete files from storage.'}), 500

    except Exception as e:
        db.session.rollback()
        logger.error(f"모델 삭제 오류: {e}")
        return jsonify({'status': 'error', 'message': 'Failed to delete model'}), 500


@app.route('/api/portfolio/summary')
@login_required
def portfolio_summary():
    """포트폴리오 요약 정보 API"""
    user_id = current_user.get_id()
    holdings = PortfolioHolding.query.filter_by(user_id=user_id).all()
    
    tickers = [h.ticker for h in holdings]
    price_data = get_prices_from_cache_or_fetch(tickers)

    results = []
    for holding in holdings:
        current_price_info = price_data.get(holding.ticker)
        current_price = current_price_info.get('price') if current_price_info else None
        
        results.append({
            'ticker': holding.ticker,
            'quantity': holding.quantity,
            'average_cost': holding.average_cost,
            'current_price': current_price
        })

    return jsonify(results)


@app.route('/api/portfolio/transactions')
@login_required
def portfolio_transactions():
    """포트폴리오 거래 내역 API"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 15, type=int)
    user_id = current_user.get_id()

    paginated_txs = PortfolioTransaction.query.filter_by(user_id=user_id)\
        .order_by(PortfolioTransaction.transaction_date.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)

    results = [{
        'ticker': tx.ticker,
        'transaction_type': tx.transaction_type,
        'quantity': tx.quantity,
        'price': tx.price,
        'transaction_date': tx.transaction_date.isoformat() + 'Z'
    } for tx in paginated_txs.items]

    return jsonify({
        'transactions': results,
        'pagination': {
            'page': paginated_txs.page,
            'per_page': paginated_txs.per_page,
            'total_pages': paginated_txs.pages,
            'total_items': paginated_txs.total
        }
    })


@app.route('/api/portfolio/transact', methods=['POST'])
@login_required
def portfolio_transact():
    """포트폴리오에 대한 매수/매도 거래를 처리합니다."""
    data = request.get_json()
    ticker = data.get('ticker')
    quantity_str = data.get('quantity')
    price_str = data.get('price')
    transaction_type = data.get('transaction_type') # 'BUY' or 'SELL'

    if not all([ticker, quantity_str, price_str, transaction_type]):
        return jsonify({'status': 'error', 'message': 'Missing required transaction data.'}), 400
    
    try:
        quantity = float(quantity_str)
        price = float(price_str)
    except (ValueError, TypeError):
        return jsonify({'status': 'error', 'message': 'Invalid number format for quantity or price.'}), 400

    if quantity <= 0:
        return jsonify({'status': 'error', 'message': 'Quantity must be positive.'}), 400

    user_id = current_user.get_id()

    try:
        if transaction_type == 'BUY':
            holding = PortfolioHolding.query.filter_by(user_id=user_id, ticker=ticker).first()
            if holding:
                # 기존 보유 종목 매수: 평균 단가 재계산
                new_total_cost = (holding.average_cost * holding.quantity) + (price * quantity)
                new_quantity = holding.quantity + quantity
                holding.average_cost = new_total_cost / new_quantity
                holding.quantity = new_quantity
            else:
                # 신규 종목 매수
                holding = PortfolioHolding(user_id=user_id, ticker=ticker, quantity=quantity, average_cost=price)
                db.session.add(holding)

        elif transaction_type == 'SELL':
            holding = PortfolioHolding.query.filter_by(user_id=user_id, ticker=ticker).first()
            if not holding or holding.quantity < quantity:
                return jsonify({'status': 'error', 'message': 'Not enough shares to sell.'}), 400
            
            if holding.quantity - quantity < 0.0001: # 거의 모든 주식을 매도하는 경우
                db.session.delete(holding)
            else:
                holding.quantity -= quantity

        else:
            return jsonify({'status': 'error', 'message': 'Invalid transaction type.'}), 400

        # 거래 내역 기록
        new_transaction = PortfolioTransaction(
            user_id=user_id, 
            ticker=ticker, 
            transaction_type=transaction_type, 
            quantity=quantity, 
            price=price
        )
        db.session.add(new_transaction)
        db.session.commit()

        return jsonify({'status': 'success', 'message': f'{transaction_type} successful.'})

    except Exception as e:
        db.session.rollback()
        logger.error(f"거래 처리 오류: {e}")
        return jsonify({'status': 'error', 'message': 'An error occurred during the transaction.'}), 500

# --- 인증 관련 라우트들 ---

@app.route('/register', methods=['GET', 'POST'])
def register():
    """사용자 등록"""
    if request.method == 'POST':
        try:
            # 폼 데이터 검증
            user_id = request.form.get('Id', '').strip()
            password = request.form.get('Pw', '')
            name = request.form.get('Name', '').strip()
            hp = request.form.get('HP', '').strip()
            age = request.form.get('Age', '')
            email1 = request.form.get('Email_1', '').strip()
            email2 = request.form.get('Email_2', '').strip()
            email = f"{email1}@{email2}" if email1 and email2 else ""
            is_adult_checked = 'True' if request.form.get('isadult_checkbox') else 'False'

            # 필수 필드 검증
            if not all([user_id, password, name, hp, age, email1, email2]):
                flash('모든 필수 필드를 입력해주세요.', 'error')
                return render_template('register.html')

            # 아이디 중복 확인
            if UsersInfo.query.filter_by(user_id=user_id).first():
                flash('이미 존재하는 아이디입니다.', 'error')
                return render_template('register.html')

            # 비밀번호 해싱
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

            # 새 사용자 생성
            new_user = UsersInfo(
                user_id=user_id,
                password=hashed_password,
                name=name,
                hp=hp,
                age=int(age) if age else 0,
                email=email,
                isadult=is_adult_checked
            )

            db.session.add(new_user)
            db.session.commit()

            logger.info(f"새 사용자 등록: {user_id}")
            flash('회원가입이 완료되었습니다. 로그인해주세요.', 'success')
            return redirect(url_for('login'))

        except ValueError:
            flash('나이는 숫자로 입력해주세요.', 'error')
            return render_template('register.html')
        except Exception as e:
            logger.error(f"회원가입 처리 중 오류: {e}")
            flash('회원가입 처리 중 오류가 발생했습니다.', 'error')
            return render_template('register.html')

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """사용자 로그인"""
    if request.method == 'POST':
        try:
            user_id = request.form.get('Id', '').strip()
            password = request.form.get('Pw', '')

            if not user_id or not password:
                flash('아이디와 비밀번호를 입력해주세요.', 'error')
                return render_template('login.html')

            user = UsersInfo.query.filter_by(user_id=user_id).first()

            if user and check_password_hash(user.password, password):
                login_user(user)
                user.update_last_login()

                logger.info(f"사용자 로그인: {user_id}")
                flash(f'{user.name}님, 환영합니다!', 'success')

                # 로그인 후 리다이렉션
                next_page = request.args.get('next')
                return redirect(next_page) if next_page else redirect(url_for('index'))
            else:
                flash('아이디 또는 비밀번호가 올바르지 않습니다.', 'error')
                return render_template('login.html')

        except Exception as e:
            logger.error(f"로그인 처리 중 오류: {e}")
            flash('로그인 처리 중 오류가 발생했습니다.', 'error')
            return render_template('login.html')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    """사용자 로그아웃"""
    user_id = current_user.get_id()
    logout_user()
    logger.info(f"사용자 로그아웃: {user_id}")
    flash('로그아웃되었습니다.', 'info')
    return redirect(url_for('index'))


@app.route('/profile')
@login_required
def profile():
    """사용자 프로필"""
    try:
        # 사용자 통계
        user_id = current_user.get_id()
        model_count = len(glob.glob(os.path.join('models', user_id, '*.zip'))) if os.path.exists(
            os.path.join('models', user_id)) else 0
        analysis_count = ModelPerformance.query.filter_by(user_id=user_id).count()

        # 최고 성과 모델
        best_performance = ModelPerformance.query.filter_by(user_id=user_id) \
            .order_by(ModelPerformance.total_return.desc()).first()

        stats = {
            'model_count': model_count,
            'analysis_count': analysis_count,
            'best_performance': best_performance
        }

        return render_template('profile.html', stats=stats)

    except Exception as e:
        logger.error(f"프로필 로드 오류: {e}")
        flash('프로필을 불러오는 중 오류가 발생했습니다.', 'error')
        return redirect(url_for('index'))


# --- 미들웨어 및 후처리 ---

@app.before_request
def before_request():
    """요청 전처리"""
    # 정적 파일 요청은 로깅하지 않음
    if request.endpoint and 'static' not in request.endpoint:
        logger.debug(f"Request: {request.method} {request.path}")


@app.after_request
def log_user_visit(response):
    """사용자 방문 로그 기록"""
    try:
        if current_user.is_authenticated:
            # X-Forwarded-For 헤더는 'client, proxy1, proxy2' 형태일 수 있으므로, 가장 첫 번째 IP(실제 클라이언트 IP)만 저장합니다.
            forwarded_for = request.headers.get('X-Forwarded-For')
            if forwarded_for:
                # 쉼표로 구분된 IP 목록 중 첫 번째 IP를 가져와 공백을 제거합니다.
                ip_address = forwarded_for.split(',')[0].strip()
            else:
                # 헤더가 없으면 기존 방식대로 remote_addr을 사용합니다.
                ip_address = request.remote_addr

            if request.endpoint and 'static' not in request.endpoint and request.endpoint != 'log_user_visit':
                new_log = UserLogs(
                    user_id=current_user.get_id(),
                    address=request.path,
                    user_agent=request.headers.get('User-Agent'),
                    ip_address=ip_address
                )
                db.session.add(new_log)
                db.session.commit()
    except Exception as e:
        logger.error(f"방문 로그 기록 오류: {e}")
        db.session.rollback()

    return response


@app.after_request
def after_request(response):
    """응답 후처리"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response


# --- 헬퍼 함수들 ---

def init_database():
    """데이터베이스 초기화"""
    try:
        with app.app_context():
            db.create_all()
            logger.info("데이터베이스 초기화 완료")
    except Exception as e:
        logger.error(f"데이터베이스 초기화 실패: {e}")


def ensure_directories():
    """필요한 디렉토리 생성"""
    directories = ['models', 'logs', 'temp']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"디렉토리 생성: {directory}")


# --- 컨텍스트 프로세서 ---

@app.context_processor
def inject_globals():
    """템플릿 전역 변수"""
    return {
        'current_time': datetime.utcnow(),
        'app_name': 'AI Stock Trading System',
        'version': '2.0.0'
    }


# --- CLI 명령어들 ---

@app.cli.command()
def init_db():
    """데이터베이스 초기화 명령어"""
    init_database()
    click.echo('Database initialized.')


@app.cli.command()
def create_admin():
    """관리자 계정 생성"""
    import click

    admin_id = click.prompt('Admin ID')
    password = click.prompt('Password', hide_input=True, confirmation_prompt=True)
    name = click.prompt('Name')
    email = click.prompt('Email')

    try:
        hashed_password = generate_password_hash(password)
        admin = UsersInfo(
            user_id=admin_id,
            password=hashed_password,
            name=name,
            hp='010-0000-0000',
            age=30,
            email=email,
            isadult='True'
        )

        db.session.add(admin)
        db.session.commit()
        click.echo(f'Admin user {admin_id} created successfully.')

    except Exception as e:
        click.echo(f'Error creating admin: {e}')


# --- 앱 시작점 ---

if __name__ == '__main__':
    ensure_directories()
    # init_database() # 데이터베이스는 앱 컨텍스트 밖에서 직접 초기화하지 않는 것이 좋습니다.
                      # flask init-db CLI 명령어를 사용하세요.

    logger.info("=" * 50)
    logger.info("AI Stock Trading System Starting")
    logger.info("=" * 50)

    # 개발 서버 실행
    # Render와 같은 프로덕션 환경에서는 gunicorn을 사용하므로 이 부분은 로컬 개발 시에만 실행됩니다.
    # DEBUG 모드는 FLASK_DEBUG 환경 변수로 제어하는 것이 좋습니다.
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=(os.environ.get('FLASK_DEBUG', 'False').lower() == 'true')
    )