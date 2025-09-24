
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
import os
import logging
import traceback
import secrets
import json
import pkg_resources
from pykrx.stock import get_market_ticker_list, get_market_ticker_name
import pymysql
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', '127.0.0.1'),
    'user': os.environ.get('DB_USER', 'root'), # 예시 값. 실제 값으로 변경
    'password': os.environ.get('DB_PASSWORD', '1234'), # 예시 값. 실제 값으로 변경
    'database': os.environ.get('DB_NAME', 'my_stock'), # 예시 값. 실제 값으로 변경
    'charset': 'utf8mb4'
}

# 분리된 모듈 임포트
from analysis import StockAnalyzer
from training import run_training_process

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = secrets.token_urlsafe(16)

# 전역 분석기 인스턴스
analyzer = StockAnalyzer()

@app.route('/')
def index():
    """메인 페이지 렌더링"""
    return render_template('index.html')

@app.route('/models', methods=['GET'])
def get_models():
    """'models' 디렉토리에 있는 모델 파일 목록을 반환합니다."""
    models_dir = 'models'
    try:
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        models = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
        return jsonify(models)
    except Exception as e:
        logger.error(f"모델 목록을 가져오는 중 오류: {e}")
        return jsonify({'error': '모델 목록을 가져올 수 없습니다.'}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """주식 분석 요청을 처리합니다."""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper().strip()
        model_name = data.get('model_name')

        if not symbol or not model_name:
            return jsonify({'error': '종목 코드와 모델을 모두 선택해주세요.'}), 400

        if not analyzer.load_model(model_name):
            return jsonify({'error': f'{model_name} 모델을 로드할 수 없습니다.'}), 500

        result = analyzer.analyze_stock(
            symbol=symbol,
            period=data.get('period', '6mo'),
            initial_balance=float(data.get('initial_balance', 10000))
        )
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"분석 API 오류: {e}")
        traceback.print_exc()
        return jsonify({'error': f'서버 오류가 발생했습니다: {str(e)}'}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """모델 훈련을 시작하고 진행 상황을 스트리밍합니다."""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper().strip()
        period = data.get('period', '1y')

        if not symbol:
            return jsonify({'error': '훈련할 종목 코드를 입력해주세요.'}), 400

        # 스트리밍 응답을 위해 제너레이터 함수를 사용
        return Response(stream_with_context(run_training_process(symbol, period)), content_type='application/json')

    except Exception as e:
        logger.error(f"훈련 API 오류: {e}")
        traceback.print_exc()
        return jsonify({'error': f'훈련 중 서버 오류가 발생했습니다: {str(e)}'}), 500

@app.route('/search_korean_stock')
def search_korean_stock():
    query = request.args.get('query', '').upper()
    if not query:
        return jsonify([])

    all_tickers = []
    for market in ["KOSPI", "KOSDAQ"]:
        tickers = get_market_ticker_list(market=market)
        for ticker in tickers:
            all_tickers.append({"ticker": ticker, "name": get_market_ticker_name(ticker)})

    results = [stock for stock in all_tickers if query in stock['name'].upper() or query in stock['ticker']]
    return jsonify(results[:10]) # Return top 10 results

@app.route('/chart')
def chart():
    """상세 차트 페이지 렌더링"""
    return render_template('chart.html')
@app.route('/login')
def login():

    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
