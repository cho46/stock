from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response, stream_with_context
from training import run_training_process
from analysis import StockAnalyzer
import os
import glob
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# 1. Flask 앱 및 기본 설정
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_very_secret_key'

# 2. 데이터베이스 설정 (MySQL)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:1234@127.0.0.1:3307/my_stock'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# 3. Flask-Login 설정
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = '/login'

# 4. 사용자 모델 정의
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

    def get_id(self):
        return self.user_id

# 5. 사용자 로더 함수
@login_manager.user_loader
def load_user(user_id):
    return UsersInfo.query.get(user_id)

# --- 라우트(Routes) 정의 ---

@app.route('/train', methods=['POST'])
@login_required
def train_model_route():
    data = request.get_json()
    symbol = data.get('symbol')
    period = data.get('period')
    model_name = data.get('model_name')
    user_id = current_user.get_id()

    if not all([symbol, period, model_name, user_id]):
        return jsonify({"status": "error", "message": "모든 필드를 채워주세요."}), 400

    return Response(stream_with_context(run_training_process(symbol, period, user_id, model_name)), content_type='application/json')

@app.route('/analyze', methods=['POST'])
@login_required
def analyze_stock_route():
    data = request.get_json()
    model_name = data.get('model_name')
    symbol = data.get('symbol')
    user_id = current_user.get_id()

    if not all([model_name, symbol, user_id]):
        return jsonify({"error": "모델, 종목코드 정보가 필요합니다."}), 400

    analyzer = StockAnalyzer()
    if not analyzer.load_model(model_name, user_id):
        return jsonify({"error": "모델을 불러오는 데 실패했습니다."}), 500
    
    result = analyzer.analyze_stock(
        symbol=symbol, 
        period=data.get('period', '1y'), 
        initial_balance=float(data.get('initial_balance', 10000))
    )
    
    if 'error' in result:
        return jsonify(result), 500

    return jsonify(result)

@app.route('/search_korean_stock')
def search_stock_route():
    # This is a placeholder. You might need a real implementation.
    # from utils import search_korean_stock 
    # query = request.args.get('query', '')
    # results = search_korean_stock(query)
    # return jsonify(results)
    return jsonify([{"ticker": "005930", "name": "삼성전자"}, {"ticker": "000660", "name": "SK하이닉스"}])

@app.route('/chart')
def chart_page():
    return render_template('chart.html')

@app.route('/models')
@login_required
def get_models():
    user_id = current_user.get_id()
    user_models_path = os.path.join(os.path.dirname(__file__), 'models', user_id)
    
    if not os.path.exists(user_models_path):
        return jsonify([])

    model_files = [os.path.basename(f) for f in glob.glob(os.path.join(user_models_path, '*.zip'))]
    return jsonify(model_files)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_id = request.form.get('Id')
        password = request.form.get('Pw')
        name = request.form.get('Name')
        hp = request.form.get('HP')
        age = request.form.get('Age')
        email1 = request.form.get('Email_1')
        email2 = request.form.get('Email_2')
        email = f"{email1}@{email2}" if email1 and email2 else ""
        is_adult_checked = 'True' if request.form.get('isadult_checkbox') else 'False'

        if UsersInfo.query.get(user_id):
            flash('이미 존재하는 아이디입니다.')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

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

        flash('회원가입이 완료되었습니다. 로그인해주세요.')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        user_id = request.form.get('Id')
        password = request.form.get('Pw')
        user = UsersInfo.query.get(user_id)

        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('아이디 또는 비밀번호가 올바르지 않습니다.')
            return render_template('login.html')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('로그아웃되었습니다.')
    return redirect(url_for('index'))

@app.route('/profile')
@login_required
def profile():
    return f'안녕하세요, {current_user.name}님!'

class UserLogs(db.Model):
    __tablename__ = 'user_logs'
    visit_Seq = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String(50), nullable=False)
    address = db.Column(db.Text, nullable=False)
    time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

@app.after_request
def log_user_visit(response):
    if current_user.is_authenticated:
        if request.endpoint and 'static' not in request.endpoint:
            new_log = UserLogs(
                user_id=current_user.get_id(),
                address=request.path,
            )
            db.session.add(new_log)
            db.session.commit()
    return response

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)