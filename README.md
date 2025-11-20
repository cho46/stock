# AI 기반 주식 분석 및 트레이딩 시스템

이 프로젝트는 사용자가 자신만의 주식 분석 AI 모델을 생성하고, 백테스팅을 통해 성과를 검증하며, 가상 포트폴리오를 관리할 수 있는 웹 기반 플랫폼입니다. 강화학습(PPO)을 이용하여 특정 주식에 대한 매매 전략을 학습하고, 사용자는 이 모델을 통해 투자 결정을 내리는 데 도움을 받을 수 있습니다.

##  주요 기능

- ** AI 모델 훈련**: 원하는 주식 종목과 투자 전략(보수적, 균형적, 공격적)을 선택하여 개인화된 강화학습 모델을 훈련합니다.
- ** 상세 분석 및 백테스팅**: 훈련된 모델을 사용하여 과거 데이터 기반의 백테스팅을 실행하고, 수익률, MDD, 샤프 지수 등 다양한 성과 지표를 시각화된 차트와 함께 확인합니다.
- ** 가상 포트폴리오 관리**: 실제와 유사한 환경에서 매수/매도 거래를 기록하고, 보유 종목의 현재 가치 및 수익률을 추적합니다.
- ** 주식 탐색 및 즐겨찾기**: 미국 증시(나스닥)에 상장된 주식을 검색하고, 관심 종목을 즐겨찾기에 추가하여 관리합니다.
- ** 사용자 인증**: 안전한 회원가입 및 로그인 기능을 통해 개인의 모델과 포트폴리오 정보를 보호합니다.
- ** 클라우드 연동**: 훈련된 모델과 관련 데이터는 Azure Blob Storage에 안전하게 저장 및 관리됩니다.

##  기술 스택

- **백엔드**: Flask, Flask-SQLAlchemy, Flask-Login
- **데이터베이스**: Microsoft SQL Server
- **머신러닝**: Stable-Baselines3 (PPO), Scikit-learn, PyTorch/TensorFlow
- **데이터 처리**: Pandas, NumPy
- **데이터 수집**: yfinance
- **클라우드**: Azure Blob Storage
- **웹 프론트엔드**: HTML, CSS, JavaScript (Jinja2 템플릿 엔진 사용)

##  프로젝트 구조

```
.
├─── main_app.py              # Flask 메인 애플리케이션, 라우팅 및 API
├─── analysis.py              # 강화학습 환경(Gym) 및 백테스팅 분석기
├─── training.py              # 모델 훈련 파이프라인
├─── utils.py                 # 데이터 다운로드, 기술적 지표 계산, Azure 연동 유틸리티
├─── populate_stocks.py       # DB에 주식 목록을 채우는 스크립트
├─── requirements.txt         # Python 패키지 의존성 목록
├─── .env.example             # 필요한 환경 변수 예시 파일
├─── templates/               # 웹 페이지 HTML 템플릿
│    ├─── index.html          # 메인 페이지
│    ├─── analysis.html       # 분석 및 백테스팅 결과 페이지
│    ├─── models.html         # 사용자 모델 관리 페이지
│    └─── ...
├─── static/                  # CSS, JavaScript, 이미지 등 정적 파일
└─── models/                  # (현재 사용되지 않음, Azure로 대체)
```

##  시작하기

### 1. 환경 설정

**요구사항**: Python 3.9 이상, Microsoft ODBC Driver for SQL Server

1.  **Git 리포지토리 복제**:
    ```bash
    git clone <repository-url>
    cd stock_analyzer
    ```

2.  **가상 환경 생성 및 활성화**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate    # Windows
    ```

3.  **필수 패키지 설치**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **환경 변수 설정**:
    `.env.example` 파일을 복사하여 `.env` 파일을 생성하고, 파일 내의 값들을 실제 환경에 맞게 수정합니다.

    -   `SECRET_KEY`: Flask 애플리케이션의 시크릿 키 (강력한 무작위 문자열 추천)
    -   `DATABASE_URL`: MS SQL 데이터베이스 연결 문자열
    -   `AZURE_STORAGE_CONNECTION_STRING`: 모델 저장을 위한 Azure Blob Storage 연결 문자열

    **.env 파일 예시**:
    ```
    SECRET_KEY="your_super_secret_key"
    DATABASE_URL="mssql+pyodbc://user:password@host:port/database?driver=ODBC+Driver+17+for+SQL+Server"
    AZURE_STORAGE_CONNECTION_STRING="your_azure_storage_connection_string"
    ```

### 2. 데이터베이스 설정

1.  **데이터베이스 초기화**:
    Flask CLI를 사용하여 데이터베이스 테이블을 생성합니다.
    ```bash
    flask init-db
    ```

2.  **주식 목록 채우기**:
    `nasdaq-listed.csv` 파일이 프로젝트 루트에 있는지 확인한 후, 다음 스크립트를 실행하여 `us_stock_info` 테이블을 채웁니다.
    ```bash
    python populate_stocks.py
    ```

### 3. 애플리케이션 실행

다음 명령어를 사용하여 Flask 개발 서버를 시작합니다.
```bash
python main_app.py
```

서버가 시작되면 웹 브라우저에서 `http://127.0.0.1:5000` 로 접속할 수 있습니다.
