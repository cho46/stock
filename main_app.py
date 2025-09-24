from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import glob
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# 1. Flask 앱 및 기본 설정
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_very_secret_key'  # 실제 서비스에서는 복잡한 키로 변경하세요.

# 2. 데이터베이스 설정 (MySQL)
# 'mysql+pymysql://[사용자명]:[비밀번호]@[DB호스트]:[포트]/[DB이름]' 형식으로 작성하세요.
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:1234@127.0.0.1:3307/my_stock'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# 3. Flask-Login 설정
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = '/login'  # 로그인하지 않은 사용자가 접근 시 리디렉션할 뷰

# 4. 사용자 모델 정의 (UserMixin 상속)
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

    # Flask-Login이 사용자를 식별하기 위해 사용하는 메서드
    def get_id(self):
        return self.user_id

# 5. 사용자 로더 함수
@login_manager.user_loader
def load_user(user_id):
    return UsersInfo.query.get(user_id)

import os
import glob
from flask import jsonify

# --- 라우트(Routes) 정의 ---

@app.route('/models')
def get_models():
    models_path = os.path.join(os.path.dirname(__file__), 'models')
    model_files = [os.path.basename(f) for f in glob.glob(os.path.join(models_path, '*.zip'))]
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

        # 디버깅 코드 2: 로그인 시도 정보 출력
        print(f"\n[DEBUG] Login attempt for user_id: {user_id}")
        if user:
            print(f"[DEBUG] User found in DB: {user.user_id}")
            print(f"[DEBUG] Stored hash in DB: {user.password}")
            print(f"[DEBUG] Password from form: {password}")
            is_match = check_password_hash(user.password, password)
            print(f"[DEBUG] Hash check result: {is_match}")
        else:
            print("[DEBUG] User not found in DB.")

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
                time = datetime.now()
            )
            db.session.add(new_log)
            db.session.commit()
    return response


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
