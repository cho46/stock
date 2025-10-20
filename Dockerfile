# Python 3.11 버전을 기반으로 하는 공식 Docker 이미지를 사용합니다.
FROM python:3.11-slim-bullseye

# 작업 디렉토리를 /app으로 설정합니다.
WORKDIR /app

# Microsoft ODBC Driver 설치에 필요한 시스템 패키지들을 설치합니다.
RUN apt-get update && apt-get install -y curl gnupg

# Microsoft GPG 키를 다운로드하고, gpg를 사용하여 dearmor한 후, 키링 디렉토리에 저장합니다. (권장되는 최신 방식)
RUN curl -sSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /usr/share/keyrings/microsoft-prod.gpg

# Microsoft 패키지 리포지토리를 추가하면서, 위에서 저장한 키를 사용하도록 명시합니다.
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/11/prod bullseye main" > /etc/apt/sources.list.d/mssql-release.list

# 패키지 목록을 다시 업데이트하고, EULA에 동의하며 ODBC 드라이버와 개발 도구를 설치합니다.
RUN apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql17 unixodbc-dev

# requirements.txt 파일을 먼저 복사하여 pip 패키지를 설치합니다.
COPY requirements.txt .

# pip과 빌드 도구를 업그레이드하고 requirements.txt에 명시된 모든 패키지를 설치합니다.
RUN pip install --upgrade pip setuptools wheel && pip install -r requirements.txt

# 나머지 모든 프로젝트 코드를 /app 디렉토리로 복사합니다.
COPY . .

# Render가 앱을 실행할 때 사용할 명령어를 지정합니다.
CMD ["gunicorn", "--bind", "0.0.0.0:${PORT}", "--timeout", "600", "main_app:app"]