# train_local.py - 로컬 PC에서 모델 훈련을 시작하고 결과를 DB와 Azure에 모두 반영하는 스크립트

import os
import click
import json
import logging
from datetime import datetime

# .env 파일 로드를 가장 먼저 수행
from dotenv import load_dotenv
load_dotenv()

# Flask 앱 컨텍스트를 가져오기 위해 main_app의 app, db와 DB 모델을 import
from main_app import app, db, ModelFiles
from training import run_training_process

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@click.command()
@click.option('--symbol', prompt='Stock Symbol (e.g., AAPL)', help='The stock symbol to train on.')
@click.option('--model-name', prompt='Model Name', help='A unique name for your new model.')
@click.option('--user-id', prompt='Your User ID', help='The user ID to associate with this model.')
@click.option('--strategy', type=click.Choice(['conservative', 'balanced', 'aggressive'], case_sensitive=False), default='balanced', help='The training strategy.')
def start_local_training(symbol, model_name, user_id, strategy):
    """
    로컬에서 모델 훈련을 시작하고, 결과를 Azure Blob Storage와 애플리케이션 DB에 모두 반영합니다.
    .env 파일에서 AZURE_STORAGE_CONNECTION_STRING와 DATABASE_URL을 자동으로 읽어옵니다.
    """
    click.echo("--- 로컬 모델 훈련 시작 ---")

    # Flask 앱 컨텍스트 내에서 모든 작업을 수행
    with app.app_context():
        # 1. 환경 변수 확인 (dotenv로 로드됨)
        if not os.getenv('AZURE_STORAGE_CONNECTION_STRING') or not os.getenv('DATABASE_URL'):
            click.echo(click.style("오류: .env 파일에 AZURE_STORAGE_CONNECTION_STRING와 DATABASE_URL이 모두 설정되어야 합니다.", fg='red'))
            return

        click.echo(click.style("환경 변수 로드 완료.", fg='green'))

        # 2. training.py의 훈련 프로세스 실행
        click.echo(f"훈련 파라미터: Symbol={symbol}, Model={model_name}, User={user_id}, Strategy={strategy}")

        final_model_info = None
        training_generator = run_training_process(
            symbol=symbol.upper(),
            period='10y', # 훈련 기간은 10년으로 고정
            user_id=user_id,
            model_name=model_name,
            strategy=strategy
        )

        # 제너레이터에서 오는 진행 상황을 실시간으로 출력
        for log_json in training_generator:
            try:
                log = json.loads(log_json)
                percentage = log.get('percentage', 0)
                message = log.get('message', '')
                status = log.get('status', 'progress_update')

                # 진행률 바 표시
                bar = '#' * int(percentage / 4)
                click.echo(f"[{bar:<25}] {percentage}% - {message}")

                if status == 'error':
                    click.echo(click.style(f"오류 발생: {message}", fg='red'))
                    # 오류 발생 시 final_model_info가 세팅되지 않도록 함
                    final_model_info = None
                    break
                elif status == 'success':
                    # 성공 시 최종 모델 정보를 저장
                    final_model_info = log.get("model_info")

            except json.JSONDecodeError:
                click.echo(log_json) # JSON이 아닌 경우 그냥 출력

        # 3. 훈련이 성공적으로 끝나면 DB에 결과 저장
        if final_model_info:
            try:
                logger.info("훈련 성공. 데이터베이스에 모델 정보 저장 시작...")
                # main_app.py의 DB 저장 로직과 동일하게 구현
                model_entry = ModelFiles.query.filter_by(
                    user_id=user_id,
                    model_name=final_model_info['model_name']
                ).first()

                if model_entry:
                    logger.info(f"기존 모델 정보를 업데이트합니다: {model_entry.model_name}")
                    model_entry.symbol = final_model_info['symbol']
                    model_entry.strategy = final_model_info['strategy']
                    model_entry.training_date = datetime.now()
                    model_entry.blob_model_path = final_model_info['blob_model_path']
                    model_entry.blob_scaler_path = final_model_info['blob_scaler_path']
                    model_entry.blob_metadata_path = final_model_info['blob_metadata_path']
                    model_entry.is_deleted = False
                else:
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
                click.echo(click.style(f"성공: 모델 정보 '{final_model_info['model_name']}'가 데이터베이스에 저장되었습니다.", fg='green'))

            except Exception as e:
                db.session.rollback()
                logger.error(f"DB에 모델 정보 저장 실패: {e}")
                click.echo(click.style(f"오류: 데이터베이스에 모델 정보를 저장하는 중 문제가 발생했습니다: {e}", fg='red'))
        else:
            click.echo(click.style("훈련이 실패했거나 중단되어 데이터베이스에 저장할 모델 정보가 없습니다.", fg='yellow'))


    click.echo("--- 로컬 모델 훈련 종료 ---")

if __name__ == '__main__':
    start_local_training()