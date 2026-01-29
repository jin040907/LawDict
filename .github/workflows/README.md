# GitHub Actions 워크플로우 가이드

이 디렉토리에는 LawDict 프로젝트의 CI/CD 파이프라인 설정이 포함되어 있습니다.

## 워크플로우 파일

### 1. `ci.yml` - 지속적 통합 (CI)
- **트리거**: `main`, `master`, `develop` 브랜치에 push 또는 PR 생성 시
- **기능**:
  - Python 환경 설정
  - 의존성 설치
  - 코드 린트 체크 (flake8)
  - FastAPI 앱 임포트 테스트
  - PostgreSQL 서비스 테스트

### 2. `deploy.yml` - 배포
- **트리거**: `main`, `master` 브랜치에 push 또는 수동 실행
- **기능**:
  - 프로덕션 배포 스크립트 실행
  - 실제 서버 배포 (SSH, Docker, 클라우드 등)

## GitHub Secrets 설정

GitHub Actions에서 사용할 시크릿을 설정해야 합니다:

1. GitHub 저장소로 이동
2. **Settings** > **Secrets and variables** > **Actions** 클릭
3. **New repository secret** 클릭하여 다음 시크릿 추가:

### 필수 시크릿

- `OPENAI_API_KEY`: OpenAI API 키
- `DB_URL`: PostgreSQL 데이터베이스 연결 문자열
  - 형식: `postgresql://username:password@host:port/database`

### 선택적 시크릿

- `DEPLOY_HOST`: 배포 서버 호스트
- `DEPLOY_USER`: 배포 서버 사용자명
- `DEPLOY_SSH_KEY`: SSH 개인 키 (배포용)

## 로컬호스트 vs GitHub Actions

### 로컬호스트 설정
현재 프로젝트는 로컬호스트에서 실행되도록 설정되어 있습니다:
- `main.py`의 `DB_URL`이 하드코딩되어 있음
- 포트 8888 사용

### GitHub Actions에서 실행하기
GitHub Actions에서는:
- 환경 변수를 통해 설정을 주입합니다
- PostgreSQL 서비스를 컨테이너로 실행합니다
- 테스트용 데이터베이스를 자동으로 생성합니다

## 환경 변수 사용 방법

`service/main.py`에서 하드코딩된 DB_URL을 환경 변수로 변경하는 것을 권장합니다:

```python
# 기존 코드
DB_URL = "postgresql://cginside19:1234@localhost:5432/bill_db"

# 변경 후
DB_URL = os.getenv("DB_URL", "postgresql://cginside19:1234@localhost:5432/bill_db")
```

이렇게 하면:
- 로컬: 기본값 사용
- GitHub Actions: 환경 변수에서 가져옴
- 프로덕션: 환경 변수로 설정

## 워크플로우 커스터마이징

### 테스트 추가
`ci.yml`에 실제 테스트를 추가할 수 있습니다:

```yaml
- name: Run tests
  run: |
    pytest tests/
```

### 배포 스크립트 수정
`deploy.yml`에 실제 배포 로직을 추가하세요:

```yaml
- name: Deploy via SSH
  uses: appleboy/ssh-action@master
  with:
    host: ${{ secrets.DEPLOY_HOST }}
    username: ${{ secrets.DEPLOY_USER }}
    key: ${{ secrets.DEPLOY_SSH_KEY }}
    script: |
      cd /path/to/lawdict
      git pull
      source venv/bin/activate
      pip install -r requirements.txt
      systemctl restart lawdict
```

## 문제 해결

### PostgreSQL 연결 오류
- GitHub Actions의 PostgreSQL 서비스가 정상적으로 시작되었는지 확인
- 포트 매핑이 올바른지 확인 (기본값: 5432)

### OpenAI API 오류
- GitHub Secrets에 `OPENAI_API_KEY`가 올바르게 설정되었는지 확인
- API 사용량 한도 확인

### 의존성 설치 오류
- `requirements.txt`의 패키지 버전 확인
- Python 버전 호환성 확인 (3.10 이상)
