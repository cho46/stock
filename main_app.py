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
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import or_
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask 앱 및 기본 설정
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_very_secret_key_change_in_production'
app.config['JSON_AS_ASCII'] = False  # 한글 JSON 지원

# 데이터베이스 설정 (MySQL)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:1234@127.0.0.1:3307/my_stock'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_recycle': 3600,
    'pool_pre_ping': True
}

db = SQLAlchemy(app)

# Flask-Login 설정
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = '/login'
login_manager.login_message = '로그인이 필요합니다.'
login_manager.login_message_category = 'warning'


# 사용자 모델 정의 (기존 DB 스키마에 맞게 수정)
class UsersInfo(UserMixin, db.Model):
    __tablename__ = 'users_info'
    users_seq = db.Column(db.Integer, autoincrement=True, index=True)
    user_id = db.Column(db.String(50), primary_key=True, unique=True, nullable=False)
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



# 사용자 로더 함수
@login_manager.user_loader
def load_user(user_id):
    return UsersInfo.query.get(user_id)


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
    """개선된 모델 훈련 API"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "요청 데이터가 없습니다."}), 400

        # 필수 파라미터 검증
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

        # 입력값 검증
        valid_periods = ['3m', '6m', '1y', '2y', '5y', '10y']
        if period not in valid_periods:
            return jsonify({
                "status": "error",
                "message": f"유효하지 않은 기간입니다. 사용 가능한 기간: {', '.join(valid_periods)}"
            }), 400

        valid_strategies = ['conservative', 'balanced', 'aggressive']
        if strategy not in valid_strategies:
            return jsonify({
                "status": "error",
                "message": f"유효하지 않은 전략입니다. 사용 가능한 전략: {', '.join(valid_strategies)}"
            }), 400

        logger.info(f"훈련 시작 - 사용자: {user_id}, 종목: {symbol}, 전략: {strategy}")

        # 스트리밍 응답으로 훈련 진행 상황 전달
        return Response(
            stream_with_context(run_training_process(symbol, period, user_id, model_name, strategy)),
            content_type='application/json; charset=utf-8'
        )

    except Exception as e:
        logger.error(f"훈련 요청 처리 중 오류: {e}")
        return jsonify({"status": "error", "message": "훈련 요청 처리 중 오류가 발생했습니다."}), 500


@app.route('/analyze', methods=['POST'])
@login_required
def analyze_stock_route():
    """개선된 주식 분석 API"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "요청 데이터가 없습니다."}), 400

        # 필수 파라미터 검증
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

        # 분석기 생성 및 모델 로드
        analyzer = StockAnalyzer()
        if not analyzer.load_model(model_name, user_id):
            return jsonify({"error": "모델을 불러오는 데 실패했습니다. 모델 파일을 확인해주세요."}), 500

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
            logger.warning(f"성과 기록 저장 실패: {e}")
            # 성과 기록 실패는 주요 기능에 영향을 주지 않으므로 계속 진행

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
    """사용자 모델 목록 조회 API"""
    try:
        user_id = current_user.get_id()
        user_models_path = os.path.join("D:\\", "StockModelFolder", user_id)

        if not os.path.exists(user_models_path):
            logger.info(f"모델 디렉토리가 존재하지 않음: {user_models_path}")
            return jsonify([])

        model_files = []
        zip_files = glob.glob(os.path.join(user_models_path, '*.zip'))

        logger.info(f"찾은 모델 파일 수: {len(zip_files)}")

        for file_path in zip_files:
            try:
                model_name = os.path.basename(file_path)

                # 메타데이터 읽기 시도
                metadata_path = file_path.replace('.zip', '_metadata.json')
                metadata = {}

                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                    except json.JSONDecodeError as je:
                        logger.warning(f"메타데이터 JSON 파싱 실패 ({model_name}): {je}")
                    except Exception as e:
                        logger.warning(f"메타데이터 읽기 실패 ({model_name}): {e}")

                # 모델 정보 구성
                model_info = {
                    'filename': model_name,
                    'symbol': metadata.get('symbol', 'Unknown'),
                    'strategy': metadata.get('strategy', 'balanced'),
                    'created_date': metadata.get('training_date', 'Unknown'),
                    'expected_return': metadata.get('final_return', 0)
                }
                model_files.append(model_info)

            except Exception as e:
                logger.error(f"모델 파일 처리 중 오류 ({file_path}): {e}")
                continue

        # 생성일 기준 정렬 (정렬 실패해도 계속 진행)
        try:
            model_files.sort(key=lambda x: x.get('created_date', ''), reverse=True)
        except Exception as e:
            logger.warning(f"모델 정렬 실패: {e}")

        logger.info(f"반환할 모델 수: {len(model_files)}")
        return jsonify(model_files)

    except Exception as e:
        logger.error(f"모델 목록 조회 중 치명적 오류: {e}")
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
    """메인 페이지의 시장 현황 데이터 API (즐겨찾기 기반)"""
    import yfinance as yf
    import pandas as pd

    if not current_user.is_authenticated:
        return jsonify([])

    user_id = current_user.get_id()
    favorites = UserFavorites.query.filter_by(user_id=user_id).all()
    
    if not favorites:
        return jsonify([])

    tickers = [fav.ticker for fav in favorites]
    market_data = []

    if tickers:
        try:
            yf_data = yf.download(tickers, period='2d', progress=False)
            if not yf_data.empty:
                for ticker_symbol in tickers:
                    try:
                        # 단일 티커 조회 시에도 yf_data['Close']는 Series가 됨
                        hist = yf_data['Close'] if len(tickers) == 1 else yf_data['Close'][ticker_symbol]
                        
                        if len(hist) >= 2 and not hist.isnull().all():
                            last_price = hist.iloc[-1]
                            prev_close = hist.iloc[-2]

                            if pd.isna(last_price) or pd.isna(prev_close):
                                raise ValueError("Price data contains NaN")

                            change = last_price - prev_close
                            percent_change = (change / prev_close) * 100 if prev_close != 0 else 0
                            
                            stock_info = UsStockInfo.query.get(ticker_symbol)
                            name = stock_info.name if stock_info else ticker_symbol

                            market_data.append({
                                'ticker': ticker_symbol,
                                'name': name,
                                'price': last_price,
                                'change': change,
                                'percent_change': percent_change
                            })
                        else:
                            raise ValueError("Not enough data")

                    except Exception as e:
                        logger.warning(f'Market overview: {ticker_symbol} 데이터 처리 실패: {e}')
                        # 실패 시에도 프론트엔드 렌더링이 깨지지 않도록 None으로 채움
                        stock_info = UsStockInfo.query.get(ticker_symbol)
                        name = stock_info.name if stock_info else ticker_symbol
                        market_data.append({
                            'ticker': ticker_symbol,
                            'name': name,
                            'price': None,
                            'change': None,
                            'percent_change': None
                        })

        except Exception as e:
            logger.error(f"시장 현황 데이터 조회 실패 (일괄): {e}")

    return jsonify(market_data)


@app.route('/api/stocks_list')
@login_required
def stocks_list():
    """전체 종목 리스트 API (페이지네이션, 검색, 실시간 가격 포함)"""
    import yfinance as yf
    import pandas as pd
    import numpy as np

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
        
        paginated_stocks = query.paginate(page=page, per_page=per_page, error_out=False)
        db_stocks = paginated_stocks.items
        tickers = [stock.ticker for stock in db_stocks]

        user_favorites = UserFavorites.query.filter_by(user_id=current_user.get_id()).all()
        favorite_tickers = {fav.ticker for fav in user_favorites}

        price_data = {}
        if tickers:
            yf_data = yf.download(tickers, period='2d', progress=False)
            if not yf_data.empty:
                for ticker in tickers:
                    try:
                        hist = yf_data['Close'][ticker]
                        if len(hist) >= 2 and not hist.isnull().all():
                            last_price = hist.iloc[-1]
                            prev_close = hist.iloc[-2]
                            
                            # NaN 값 체크 및 변환
                            if pd.isna(last_price) or pd.isna(prev_close):
                                price_data[ticker] = {'price': None, 'change': None, 'percent_change': None}
                                continue

                            change = last_price - prev_close
                            percent_change = (change / prev_close) * 100 if prev_close != 0 else 0
                            
                            price_data[ticker] = {
                                'price': last_price,
                                'change': change,
                                'percent_change': percent_change
                            }
                        else:
                            price_data[ticker] = {'price': None, 'change': None, 'percent_change': None}
                    except (KeyError, IndexError):
                        price_data[ticker] = {'price': None, 'change': None, 'percent_change': None}
                        logger.warning(f'yfinance에서 {ticker} 데이터를 처리할 수 없습니다.')

        results = []
        for stock in db_stocks:
            data = price_data.get(stock.ticker, {'price': None, 'change': None, 'percent_change': None})
            results.append({
                'ticker': stock.ticker,
                'name': stock.name,
                'price': data['price'],
                'change': data['change'],
                'percent_change': data['percent_change'],
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


@app.route('/stocks')
@login_required
def stocks_page():
    """종목보기 페이지"""
    return render_template('stocks.html')


@app.route('/dashboard')
@login_required
def dashboard():
    """사용자 대시보드"""
    try:
        user_id = current_user.get_id()

        # 최근 모델 성과
        recent_performances = ModelPerformance.query.filter_by(user_id=user_id) \
            .order_by(ModelPerformance.analysis_date.desc()).limit(5).all()

        # 통계
        total_analyses = ModelPerformance.query.filter_by(user_id=user_id).count()
        avg_return = db.session.query(db.func.avg(ModelPerformance.total_return)) \
                         .filter_by(user_id=user_id).scalar() or 0

        stats = {
            'total_analyses': total_analyses,
            'avg_return': avg_return,
            'recent_performances': recent_performances
        }

        return render_template('dashboard.html', stats=stats)

    except Exception as e:
        logger.error(f"대시보드 로드 오류: {e}")
        flash('대시보드를 불러오는 중 오류가 발생했습니다.', 'error')
        return redirect(url_for('index'))


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
            if UsersInfo.query.get(user_id):
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

            user = UsersInfo.query.get(user_id)

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
            if request.endpoint and 'static' not in request.endpoint and request.endpoint != 'log_user_visit':
                new_log = UserLogs(
                    user_id=current_user.get_id(),
                    address=request.path,
                    user_agent=request.headers.get('User-Agent'),
                    ip_address=request.headers.get('X-Forwarded-For', request.remote_addr)
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


# --- 개발/프로덕션 환경 설정 ---

def configure_app_for_environment():
    """환경별 앱 설정"""
    if os.environ.get('FLASK_ENV') == 'production':
        # 프로덕션 설정
        app.config['DEBUG'] = False
        app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'production-secret-key')
        logging.getLogger().setLevel(logging.WARNING)
    else:
        # 개발 설정
        app.config['DEBUG'] = True
        logging.getLogger().setLevel(logging.DEBUG)


# --- 앱 시작점 ---

if __name__ == '__main__':
    configure_app_for_environment()
    ensure_directories()
    init_database()

    logger.info("=" * 50)
    logger.info("AI Stock Trading System Starting")
    logger.info("=" * 50)

    # 개발 서버 실행
    app.run(
        debug=app.config['DEBUG'],
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        threaded=True
    )