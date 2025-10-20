# Python 3.11 버전을 기반으로 하는 공식 Docker 이미지를 사용합니다.
FROM python:3.11-slim

# 작업 디렉토리를 /app으로 설정합니다.
WORKDIR /app

# Microsoft ODBC Driver 17 for SQL Server 설치에 필요한 패키지들을 설치합니다.
# 이 과정이 Render 서버에 드라이버를 설치하는 핵심 단계입니다.
RUN apt-get update && apt-get install -y curl gnupg
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list
RUN apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql17 unixodbc-dev

# requirements.txt 파일을 먼저 복사하여 pip 패키지를 설치합니다.
# 이렇게 하면 코드 변경 시 매번 패키지를 새로 설치하지 않아 빌드 속도가 향상됩니다.
COPY requirements.txt .

# pip과 빌드 도구를 업그레이드하고 requirements.txt에 명시된 모든 패키지를 설치합니다.
RUN pip install --upgrade pip setuptools wheel && pip install -r requirements.txt

# 나머지 모든 프로젝트 코드를 /app 디렉토리로 복사합니다.
COPY . .

# Render가 앱을 실행할 때 사용할 명령어를 지정합니다.
# Render는 PORT 환경 변수를 자동으로 제공합니다.
CMD ["gunicorn", "--bind", "0.0.0.0:${PORT}", "--timeout", "600", "main_app:app"]
