
import pandas as pd
from sqlalchemy import create_engine, text, Column, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import logging
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 설정 --- #
DB_URI = 'mysql+pymysql://root:1234@127.0.0.1:3307/my_stock'
TABLE_NAME = 'us_stock_info'
# 로컬 파일 이름
NASDAQ_FILENAME = 'nasdaq-listed.csv'

# 데이터베이스 테이블과 매칭되는 모델 정의
Base = declarative_base()
class UsStockInfo(Base):
    __tablename__ = TABLE_NAME
    ticker = Column(String(20), primary_key=True, nullable=False)
    name = Column(String(255), nullable=False)
    exchange = Column(String(20), nullable=False)

def read_local_stock_file():
    """로컬 CSV 파일에서 주식 목록을 읽고 파싱합니다."""
    if not os.path.exists(NASDAQ_FILENAME):
        logging.error(f"파일을 찾을 수 없습니다: '{NASDAQ_FILENAME}'. 먼저 파일을 프로젝트 폴더에 다운로드하세요.")
        return None

    logging.info(f"'{NASDAQ_FILENAME}' 파일에서 데이터를 읽는 중...")
    # CSV 파일이므로 read_csv 사용, 첫 줄을 헤더로 자동 인식
    df = pd.read_csv(NASDAQ_FILENAME)
    
    # User-provided file has different column names than the .txt files
    # It seems to be a direct export. Let's adapt to the columns found.
    # Expected columns: Symbol, Company Name, ...
    df.rename(columns={"Symbol": "ticker", "Security Name": "name"}, inplace=True)

    # 필요한 컬럼만 선택하고, 테스트 종목(Test Issue)은 제외
    if 'Test Issue' in df.columns and df['Test Issue'].dtype == 'object':
        df = df[df['Test Issue'] == 'N']

    df['exchange'] = 'NASDAQ'
    
    # 최종적으로 DB 테이블과 일치하는 컬럼만 선택
    final_df = df[["ticker", "name", "exchange"]]
    
    # 중복 및 불필요한 데이터 제거
    final_df.dropna(subset=['ticker', 'name'], inplace=True)
    final_df = final_df[~final_df['ticker'].str.contains('\$')]
    final_df.drop_duplicates(subset=['ticker'], keep='first', inplace=True)
    
    logging.info(f"총 {len(final_df)}개의 고유한 나스닥 종목을 찾았습니다.")
    return final_df

def populate_database(df):
    """데이터프레임의 내용을 데이터베이스에 삽입합니다."""
    if df is None or df.empty:
        logging.warning("데이터베이스에 저장할 데이터가 없습니다.")
        return

    engine = create_engine(DB_URI)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        logging.info(f"기존 테이블 '{TABLE_NAME}'의 데이터를 삭제합니다...")
        session.execute(text(f"TRUNCATE TABLE {TABLE_NAME}"))
        
        logging.info(f"{len(df)}개의 종목을 데이터베이스에 삽입합니다...")
        # 사용자의 코드에서 제안한 효율적인 merge 방식 사용
        for _, row in df.iterrows():
            ticker_obj = UsStockInfo(
                ticker=row["ticker"],
                name=row["name"],
                exchange=row["exchange"]
            )
            session.merge(ticker_obj)

        session.commit()
        logging.info("데이터베이스 채우기 완료!")

    except Exception as e:
        logging.error(f"데이터베이스 작업 중 오류 발생: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == '__main__':
    logging.info("로컬 파일로부터 주식 목록 데이터베이스 채우기 스크립트를 시작합니다.")
    stock_df = read_local_stock_file()
    if stock_df is not None:
        populate_database(stock_df)
    else:
        logging.error("주식 목록을 가져오지 못해 데이터베이스 작업을 수행할 수 없습니다.")
