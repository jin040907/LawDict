# LawDict

<div align="center">

**AI 기반 입법 분석 플랫폼**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-12+-blue.svg)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)

</div>

## 프로젝트 소개

LawDict는 머신러닝과 AI를 활용하여 법안의 처리 결과를 예측하고, 상세한 분석 리포트를 제공하는 웹 애플리케이션입니다.

### 프로젝트 목적

- **법안 처리 결과 예측**: 머신러닝 모델을 통해 법안의 최종 처리 결과를 사전에 예측
- **데이터 기반 분석**: 과거 법안 데이터를 분석하여 예측 근거 제공
- **AI 기반 리포트 생성**: 법안의 핵심 정보, 유사 사례, 사회적 영향 등을 종합 분석
- **실시간 챗봇 지원**: 법안 분석 내용에 대한 질문에 즉시 답변

### 주요 기능

- **법안 검색 및 조회**: 법안명, 발의자명, 발의일을 기준으로 법안 검색
- **AI 예측 분석**: CatBoost 기반 머신러닝 모델을 통한 법안 처리 결과 예측 및 확률 제공
- **시각화 대시보드**: 예측 확률을 도넛 차트로 직관적으로 표시
- **상세 분석 리포트**: 7개 섹션으로 구성된 종합 리포트 (법안 정보, 예측 결과, 근거 분석, 유사 사례 비교 등)
- **AI 챗봇**: OpenAI GPT 모델 기반 실시간 질의응답 시스템
- **뉴스 연동**: 유사 법안 관련 최신 뉴스 자동 수집 및 표시

## 📁 프로젝트 구조

```
LawDict/
├── data/                             # 법안 데이터 수집·통합
│   ├── collecting_bill_data_final.py # 의안 수집 스크립트 (국회 API·크롤링)
│   └── assemble_data_final.py       # 수집 데이터 통합·적재 스크립트
│
├── service/                          # FastAPI 웹 서버
│   ├── main.py                      # 메인 애플리케이션 파일
│   ├── .env                         # 환경 변수 설정 파일 (gitignore)
│   ├── static/images/               # 정적 자원 (챗봇 아이콘 등)
│   └── templates/                   # HTML 템플릿
│       ├── index.html               # 법안 목록 페이지 (검색, 필터링)
│       └── detail.html              # 법안 상세 페이지 (리포트, 챗봇)
│
├── model/                            # 머신러닝 모델 관련
│   ├── model_final.ipynb            # 모델 학습/평가 노트북
│   └── thresholds_8class_final.json # 모델 임계값 설정 파일
│
├── report/                           # 리포트 생성 관련
│   ├── ai_report_column.ipynb       # AI 리포트 생성 노트북
│   ├── news_column.ipynb            # 뉴스 수집 노트북
│   ├── news_column_gpt.ipynb        # GPT 기반 뉴스 수집 노트북
│   └── codes/                       # 리포트 생성 모듈
│       ├── generate_sample_report_llm.py
│       ├── llm_report_enricher.py
│       ├── make_sample_input.py
│       ├── model_runtime.py
│       ├── report_builder_with_llm.py
│       ├── report_insights.py
│       └── similar_cases_pgvector.py
│
├── .github/workflows/                # GitHub Actions 워크플로우
│   ├── ci.yml                        # CI 파이프라인 (자동 테스트)
│   ├── preview.yml                   # 미리보기 워크플로우 (ngrok)
│   └── deploy.yml                    # 배포 워크플로우
├── requirements.txt                  # Python 의존성 목록
├── .gitignore                        # Git 제외 파일 목록
└── README.md                         # 프로젝트 문서
```

### 디렉토리 설명

- **data/**: 법안 데이터 수집 및 통합

  - `collecting_bill_data_final.py`: 국회 API·크롤링 기반 의안 수집, 임베딩(bge-m3)·pgvector 적재
  - `assemble_data_final.py`: 수집 데이터 통합 및 학습/서비스용 테이블 적재
- **service/**: FastAPI 기반 웹 서버 코드

  - `main.py`: 모든 API 엔드포인트와 비즈니스 로직 포함
  - `templates/`: Jinja2 템플릿으로 렌더링되는 HTML 페이지
  - `static/images/`: 챗봇 아이콘 등 정적 자원
- **model/**: 머신러닝 모델 학습 및 평가

  - CatBoost 분류 모델 사용
  - 법안 처리 결과 8개 클래스 예측
- **report/**: 리포트 생성 및 데이터 처리

  - AI 리포트 자동 생성 (`ai_report_column.ipynb`, `codes/` 모듈)
  - 뉴스 데이터 수집 (`news_column.ipynb`, `news_column_gpt.ipynb`)

## 기술 스택

### Backend

- **FastAPI** (0.100+): 고성능 비동기 Python 웹 프레임워크
- **PostgreSQL** (12+): 관계형 데이터베이스
- **SQLAlchemy**: Python ORM 및 데이터베이스 툴킷
- **Pandas**: 데이터 분석 및 처리
- **Uvicorn**: ASGI 서버
- **Python-dotenv**: 환경 변수 관리

### AI/ML

- **OpenAI API**: GPT-4o-mini 기반 챗봇 및 리포트 생성
- **CatBoost**: 그래디언트 부스팅 머신러닝 모델
- **scikit-learn**: 머신러닝 유틸리티
- **NumPy**: 수치 계산

### Frontend

- **Tailwind CSS**: 유틸리티 기반 CSS 프레임워크
- **Chart.js**: 데이터 시각화 (도넛 차트)
- **Jinja2**: Python 템플릿 엔진
- **Vanilla JavaScript**: 클라이언트 사이드 로직

### 개발 도구

- **Jupyter Notebook**: 모델 학습 및 데이터 분석
- **DuckDuckGo Search**: 웹 검색 기능 (챗봇 보조)
- **GitHub Actions**: CI/CD 파이프라인

### 데이터베이스 스키마

- **PostgreSQL**: 법안 데이터 저장
- 주요 테이블: `public.final_training_data_copy_sample10_md`

## 🚀 설치 및 실행

### 사전 요구사항

- **Python**: 3.10 이상
- **PostgreSQL**: 12 이상
- **OpenAI API Key**: 챗봇 및 리포트 생성용 (필수)
- **Git**: 저장소 클론용

### 1. 저장소 클론

```bash
git clone <repository-url>
cd LawDict
```

### 2. 가상 환경 생성 및 활성화

**Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**PowerShell:**

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. 의존성 설치

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**주요 패키지 목록:**

```
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
sqlalchemy>=2.0.0
pandas>=2.0.0
python-dotenv>=1.0.0
openai>=1.0.0
markdown>=3.4.0
duckduckgo-search>=3.9.0
jinja2>=3.1.0
psycopg2-binary>=2.9.0
```

### 4. 환경 변수 설정

`service/.env` 파일을 생성하고 다음 내용을 추가하세요:

```env
# OpenAI API 설정 (필수)
OPENAI_API_KEY=sk-your-openai-api-key-here

# OpenAI 모델 선택 (선택사항, 기본값: gpt-4o-mini)
OPENAI_MODEL_PRIMARY=gpt-4o-mini
```

**환경 변수 설명:**

- `OPENAI_API_KEY`: OpenAI API 키 (https://platform.openai.com/api-keys 에서 발급)
- `OPENAI_MODEL_PRIMARY`: 사용할 OpenAI 모델 (기본값: gpt-4o-mini)

### 5. 데이터베이스 설정

#### 5.1 PostgreSQL 설치 및 데이터베이스 생성

**PostgreSQL 설치:**

- Linux: `sudo apt-get install postgresql` (Ubuntu/Debian)
- macOS: `brew install postgresql`
- Windows: https://www.postgresql.org/download/windows/

**데이터베이스 생성:**

```sql
CREATE DATABASE bill_db;
CREATE USER your_username WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE bill_db TO your_username;
```

#### 5.2 데이터베이스 연결 설정

데이터베이스 연결은 환경 변수 `DB_URL`을 통해 설정됩니다. `service/.env` 파일에 추가하거나 환경 변수로 설정할 수 있습니다:

**방법 1: .env 파일 사용 (로컬 개발)**

`service/.env` 파일에 추가:

```env
DB_URL=postgresql://username:password@localhost:5432/bill_db
```

**방법 2: 환경 변수로 설정**

```bash
export DB_URL=postgresql://username:password@localhost:5432/bill_db
```

**방법 3: Render.com 사용 (클라우드 데이터베이스)**

1. https://render.com 에서 PostgreSQL 데이터베이스 생성
2. External Database URL 복사
3. `service/.env` 또는 환경 변수에 설정

**연결 문자열 형식:**

```
postgresql://[사용자명]:[비밀번호]@[호스트]:[포트]/[데이터베이스명]
```

**참고:** `DB_URL` 환경 변수는 필수입니다. `service/.env` 파일에 설정하세요.

#### 5.3 테이블 생성

필요한 테이블이 이미 존재해야 합니다:

- `public.final_training_data_copy_sample10_md`: 법안 데이터 테이블

**주요 컬럼:**

- `bill_id` (PK): 법안 고유 ID
- `bill_name`: 법안명
- `proposer_name`: 발의자명
- `propose_dt`: 발의일
- `ai_prediction`: AI 예측 결과
- `ai_probability`: AI 예측 확률
- `ai_report`: AI 요약 리포트
- `report_md`: 상세 분석 리포트 (마크다운)
- `news`: 관련 뉴스 데이터 (JSON)

### 6. 서버 실행

#### 개발 모드 (자동 리로드)

```bash
cd service
python main.py
```

또는 uvicorn을 직접 사용:

```bash
cd service
uvicorn main:app --host 0.0.0.0 --port 8888 --reload
```

#### 프로덕션 모드

```bash
cd service
uvicorn main:app --host 0.0.0.0 --port 8888 --workers 4
```

**서버 접속:**

- 로컬: http://localhost:8888
- 네트워크: http://[서버IP]:8888

### 7. 실행 확인

서버가 정상적으로 실행되면 다음 메시지가 표시됩니다:

```
INFO:     Uvicorn running on http://0.0.0.0:8888
INFO:     Application startup complete.
```

브라우저에서 `http://localhost:8888`로 접속하여 법안 목록 페이지가 표시되는지 확인하세요.

### 문제 해결

#### 포트가 이미 사용 중인 경우

```bash
# 포트 사용 프로세스 확인
lsof -i :8888  # macOS/Linux
netstat -ano | findstr :8888  # Windows

# 다른 포트로 실행
uvicorn main:app --port 8000
```

#### 데이터베이스 연결 오류

- PostgreSQL 서비스가 실행 중인지 확인
- 연결 정보(호스트, 포트, 사용자명, 비밀번호) 확인
- 방화벽 설정 확인

#### OpenAI API 오류

- API 키가 올바른지 확인
- API 사용량 한도 확인
- 네트워크 연결 확인

## 📖 사용 방법

### 법안 검색

1. **메인 페이지 접속**: `http://localhost:8888`로 접속
2. **법안 목록 확인**: 최신순으로 정렬된 법안 목록 확인
3. **검색 기능**:
   - 검색창에 법안명 또는 발의자명 입력
   - 실시간 검색 (300ms 디바운싱)
   - Enter 키로 즉시 검색
4. **발의일 필터링**:
   - 연도 선택: 2008년 ~ 2026년
   - 월 선택: 1월 ~ 12월
   - 일 선택: 1일 ~ 31일
   - 여러 필터 조합 가능
5. **정렬 옵션**:
   - 최신순 / 오래된순
   - 법안명순 (가나다 / 역순)
   - 발의자순 (가나다 / 역순)
6. **페이지네이션**: 하단 페이지 번호로 이동

### 법안 상세 분석

1. **리포트 접근**: 법안 목록에서 "AI 리포트 보기" 버튼 클릭
2. **내비게이션**: 상단 탭으로 섹션 간 이동
3. **주요 섹션**:

   **#1 AI 모델 예측 및 분석**

   - 예측 확률 도넛 차트 (Top 3 예측 결과)
   - 모델 예측 결과 상세 설명
   - 핵심 요약 리포트

   **#2 법안 정보 및 요약**

   - 법안 요약 정보
   - 입력 법안 메타데이터

   **#3 데이터 기반 근거 분석**

   - 예측 근거 데이터 테이블
   - 피처별 기여도 분석

   **#4 유사 법안 비교 분석**

   - 유사 법안 비교 테이블
   - 유사도 점수 및 핵심 차이점
   - 관련 뉴스 (최대 10개)

   **#5 사회적 영향력 분석**

   - 법안의 사회적 영향 분석
   - 이해관계자 영향 평가

   **#6 향후 대응 방향 및 제언**

   - 대응 전략 제안
   - 권장 사항

### AI 챗봇 사용

1. 상세 페이지 우측 하단의 챗봇 버튼 클릭
2. 법안 분석 내용에 대한 질문 입력
3. 질문 예시:
   - "이 법안의 예측 확률은 얼마인가요?"
   - "주요 내용을 요약해주세요"
   - "유사한 법안은 무엇인가요?"
   - "사회적 영향은 무엇인가요?"
4. AI가 리포트 내용을 바탕으로 답변 제공
5. 스크롤하여 이전 대화 이력 확인 가능

### 고급 기능

- **빠른 미리보기**: 법안명에 마우스 오버 시 미리보기 모달 표시
- **반응형 디자인**: 모바일, 태블릿, 데스크톱 지원
- **실시간 검색**: 검색 결과 실시간 반영 (300ms 디바운싱)

## 모델 학습 및 데이터 처리

### 데이터 수집

법안 원시 데이터를 수집하고 통합합니다.

```bash
cd data
# 1) 의안 수집 (국회 API·크롤링, 임베딩·pgvector 적재)
python collecting_bill_data_final.py
# 2) 수집 데이터 통합 및 학습/서비스용 테이블 적재
python assemble_data_final.py
```

**데이터 수집 프로세스:**

1. **collecting_bill_data_final.py**: 국회 API(ALLBILL 단건조회) 및 크롤링으로 의안 수집, bge-m3 임베딩 후 pgvector(staging/integrated) 적재
2. **assemble_data_final.py**: 수집 데이터를 통합하여 학습·서비스용 최종 테이블에 적재

**필수 준비사항:**

- `ASSEMBLY_API_KEY`: 국회 API 키
- PostgreSQL에 pgvector extension 설치
- `.env` 파일에 DB 연결 정보 설정

### 모델 학습

법안 처리 결과 예측 모델을 학습합니다.

```bash
cd model
jupyter notebook model_final.ipynb
```

**모델 정보:**

- **알고리즘**: CatBoost Classifier
- **예측 클래스**: 8개 (원안가결, 수정가결, 부결, 폐기, 철회, 임기만료폐기, 대안반영폐기, 수정안반영폐기)
- **입력 피처**: 법안명, 요약, 발의일 등
- **임계값**: `thresholds_8class_final.json` 파일에 저장

**학습 프로세스:**

1. 데이터베이스에서 학습 데이터 로드
2. 피처 엔지니어링 및 전처리
3. CatBoost 모델 학습
4. 검증 데이터로 성능 평가
5. 임계값 최적화
6. 모델 저장 및 임계값 저장

### 리포트 생성

AI 리포트를 자동으로 생성합니다.

```bash
cd report
jupyter notebook ai_report_column.ipynb
```

리포트 생성 로직은 `report/codes/` 모듈(LLM 리포트 빌더, 유사 사례 pgvector 검색 등)을 사용하며, 노트북에서 호출합니다.

**리포트 생성 프로세스:**

1. 데이터베이스에서 법안 데이터 로드
2. OpenAI API를 사용하여 리포트 생성
3. 마크다운 형식으로 리포트 작성
4. 데이터베이스의 `report_md` 컬럼에 저장

**리포트 구성:**

- 0. 요약 (summary)
- 1. 입력 법안 정보 (bill_info)
- 2. 모델 예측 결과 (prediction)
- 3. 데이터 기반 근거 분석 (evidence)
- 4. 유사 사례 비교 분석 (similar_cases)
- 5. 사회적 영향력 분석 (social_impact)
- 6. 향후 대응 방향 및 제언 (next_action)

**주의사항:**

- OpenAI API 사용량 및 비용 관리 필요
- 리포트 생성에는 법안당 약 10-30초 소요
- API 레이트 리밋을 고려하여 적절한 딜레이 설정 권장

### 뉴스 수집

유사 법안 관련 뉴스를 수집합니다.

```bash
cd report
jupyter notebook news_column.ipynb
# 또는 GPT 기반 뉴스 수집
jupyter notebook news_column_gpt.ipynb
```

**뉴스 수집 프로세스:**

1. 리포트의 "유사 사례 비교 분석" 섹션에서 키워드 추출
2. 딥서치뉴스 API(또는 GPT 기반)를 사용하여 뉴스 검색
3. 발의일 기준 전후 6개월 범위로 검색
4. 최대 10개 뉴스 수집
5. JSON 형식으로 데이터베이스의 `news` 컬럼에 저장

**뉴스 데이터 구조:**

```json
[
  {
    "title": "뉴스 제목",
    "content_url": "뉴스 URL",
    "publisher": "출판사",
    "published_at": "발행일"
  }
]
```

## API 엔드포인트

### 법안 검색 API

법안을 검색하는 API입니다.

**엔드포인트:**

```
GET /api/search
```

**쿼리 파라미터:**

| 파라미터  | 타입    | 필수 | 설명                          |
| --------- | ------- | ---- | ----------------------------- |
| `q`     | string  | 선택 | 검색어 (법안명 또는 발의자명) |
| `year`  | integer | 선택 | 발의 연도 (2008-2026)         |
| `month` | integer | 선택 | 발의 월 (1-12)                |
| `day`   | integer | 선택 | 발의 일 (1-31)                |

**요청 예시:**

```bash
curl "http://localhost:8888/api/search?q=헌법&year=2024&month=1"
```

**응답 형식:**

```json
{
  "results": [
    {
      "bill_id": "12345",
      "bill_name": "법안명",
      "proposer_name": "발의자명",
      "propose_dt_str": "2024-01-01"
    }
  ],
  "count": 1
}
```

**응답 필드:**

- `results`: 검색 결과 배열
  - `bill_id`: 법안 고유 ID
  - `bill_name`: 법안명
  - `proposer_name`: 발의자명
  - `propose_dt_str`: 발의일 (YYYY-MM-DD 형식)
- `count`: 검색 결과 개수

**에러 응답:**

```json
{
  "results": [],
  "count": 0
}
```

### 법안 상세 조회 API

법안 상세 정보를 조회하는 API입니다.

**엔드포인트:**

```
GET /bill/{bill_id}
```

**경로 파라미터:**

- `bill_id`: 법안 고유 ID

**응답:** HTML 페이지 (detail.html 템플릿)

### 챗봇 API

법안 분석 내용에 대한 질문에 답변하는 API입니다.

**엔드포인트:**

```
POST /api/chat
```

**요청 헤더:**

```
Content-Type: application/json
```

**요청 본문:**

```json
{
  "message": "이 법안의 예측 확률은 얼마인가요?",
  "bill_name": "법안명",
  "context": "리포트 본문 텍스트 (최대 1500자)"
}
```

**요청 필드:**

- `message` (필수): 사용자 질문
- `bill_name` (필수): 법안명
- `context` (선택): 리포트 본문 텍스트

**요청 예시:**

```bash
curl -X POST "http://localhost:8888/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "주요 내용을 요약해주세요",
    "bill_name": "법안명",
    "context": "리포트 본문..."
  }'
```

**응답 형식:**

```json
{
  "answer": "이 법안은 ... 에 관한 내용입니다."
}
```

**응답 필드:**

- `answer`: AI가 생성한 답변 텍스트

**에러 응답:**

```json
{
  "answer": "데이터 연동 중 지연이 발생했습니다. 왼쪽 리포트의 수치를 우선 참고해 주세요!"
}
```

### 메인 페이지 API

법안 목록을 조회하는 API입니다.

**엔드포인트:**

```
GET /
GET /?page={페이지번호}
```

**쿼리 파라미터:**

- `page` (선택): 페이지 번호 (기본값: 1)

**응답:** HTML 페이지 (index.html 템플릿)

**페이지네이션:**

- 페이지당 5개 법안 표시
- 하단에 페이지 번호 표시

## 개발 참고사항

### 데이터베이스 스키마

#### 주요 테이블

**테이블명:** `public.final_training_data_copy_sample10_md`

#### 주요 컬럼

| 컬럼명             | 타입            | 설명                           |
| ------------------ | --------------- | ------------------------------ |
| `bill_id`        | VARCHAR/INTEGER | 법안 고유 ID (PK)              |
| `bill_name`      | TEXT            | 법안명                         |
| `proposer_name`  | TEXT            | 발의자명                       |
| `propose_dt`     | DATE/TIMESTAMP  | 발의일                         |
| `summary`        | TEXT            | 법안 요약                      |
| `ai_prediction`  | VARCHAR         | AI 예측 결과                   |
| `ai_probability` | DECIMAL         | AI 예측 확률 (0.0 ~ 1.0)       |
| `ai_report`      | TEXT            | AI 요약 리포트 (HTML)          |
| `report_md`      | TEXT            | 상세 분석 리포트 (마크다운)    |
| `news`           | TEXT/JSONB      | 관련 뉴스 데이터 (JSON 문자열) |
| `general_result` | VARCHAR         | 실제 처리 결과 (학습용)        |

#### 인덱스 권장사항

```sql
CREATE INDEX idx_bill_name ON public.final_training_data_copy_sample10_md(bill_name);
CREATE INDEX idx_proposer_name ON public.final_training_data_copy_sample10_md(proposer_name);
CREATE INDEX idx_propose_dt ON public.final_training_data_copy_sample10_md(propose_dt);
```

### 리포트 구조

리포트는 마크다운 형식으로 저장되며, 다음 섹션으로 구성됩니다:

#### 섹션 구성

1. **0. 요약 (summary)**

   - 법안의 핵심 요약 정보
   - 신뢰도 표시
2. **1. 입력 법안 정보 (bill_info)**

   - 법안 메타데이터
   - 발의자 정보
   - 발의일 등
3. **2. 모델 예측 결과 (prediction)**

   - 예측 확률 테이블 (순위, 예측, 확률)
   - Top 3 예측 결과
   - 예측 근거 설명
4. **3. 데이터 기반 근거 분석 (evidence)**

   - 피처별 기여도 분석
   - 예측에 영향을 미친 주요 요인
5. **4. 유사 사례 비교 분석 (similar_cases)**

   - 유사 법안 비교 테이블
   - 유사도 점수
   - 핵심 차이점
6. **5. 사회적 영향력 분석 (social_impact)**

   - 법안의 사회적 영향 평가
   - 이해관계자 영향 분석
7. **6. 향후 대응 방향 및 제언 (next_action)**

   - 대응 전략 제안
   - 권장 사항

### 코드 구조

#### 주요 함수

**`remove_report_titles()`**

- 보고서에 자동 추가된 제목 제거
- 이모지 및 마크다운 제목 패턴 제거

**`parse_prediction_table()`**

- 리포트에서 예측 확률 테이블 파싱
- Top 3 예측 결과 추출

**`split_expert_report()`**

- 전문가 보고서를 섹션별로 분리
- JSON 또는 마크다운 형식 자동 감지
- HTML 변환

**`search_web()`**

- DuckDuckGo 검색 API 사용
- 웹 검색 결과 반환

### 환경 변수

| 변수명                   | 필수 | 기본값                                             | 설명                                                                         |
| ------------------------ | ---- | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| `OPENAI_API_KEY`       | ✅   | -                                                  | OpenAI API 키                                                                |
| `OPENAI_MODEL_PRIMARY` | ❌   | `gpt-4o-mini`                                    | 사용할 OpenAI 모델                                                           |
| `DB_URL`               | ✅   | -                                                  | bill_db용 PostgreSQL 연결 문자열                                             |
| `MEMBER_DB_URL`        | ❌   | `DB_URL`에서 `/bill_db` → `/member` 로 치환 | member DB용 연결 문자열 (data/assemble_data_final.py, 유저가 다를 때만 설정) |

### 성능 최적화

- **데이터베이스 연결 풀링**: SQLAlchemy 엔진 사용
- **비동기 처리**: FastAPI 비동기 엔드포인트
- **캐싱**: 자주 조회되는 데이터 캐싱 고려
- **인덱싱**: 데이터베이스 인덱스 활용

### 보안 고려사항

- **환경 변수**: 민감한 정보는 `.env` 파일에 저장
- **SQL 인젝션 방지**: SQLAlchemy의 파라미터화된 쿼리 사용
- **XSS 방지**: HTML 이스케이프 처리
- **API 키 보호**: `.env` 파일을 `.gitignore`에 추가

### 로깅

서버 로그는 콘솔에 출력됩니다. 프로덕션 환경에서는 파일 로깅을 권장합니다:

```python
import logging
logging.basicConfig(
    filename='lawdict.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## 테스트

### 수동 테스트

1. **법안 검색 테스트**

   ```bash
   curl "http://localhost:8888/api/search?q=헌법"
   ```
2. **챗봇 테스트**

   ```bash
   curl -X POST "http://localhost:8888/api/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "테스트", "bill_name": "테스트 법안"}'
   ```
3. **페이지 접속 테스트**

   - 메인 페이지: http://localhost:8888
   - 상세 페이지: http://localhost:8888/bill/{bill_id}

### 자동화 테스트 (향후 추가)

```bash
pytest tests/
```

## 📦 배포

### GitHub Actions를 통한 테스트 및 미리보기

프로젝트에는 GitHub Actions 워크플로우가 포함되어 있습니다:

#### CI 파이프라인 (`ci.yml`)

- **트리거**: `main`, `master`, `develop` 브랜치에 push 또는 PR 생성 시
- **기능**:
  - Python 환경 설정
  - 의존성 설치
  - 코드 린트 체크
  - FastAPI 앱 임포트 테스트
  - 웹사이트 엔드포인트 테스트

#### Preview 워크플로우 (`preview.yml`)

- **트리거**: 수동 실행 (`workflow_dispatch`) 또는 PR 생성 시
- **기능**:
  - 서버 시작 및 테스트
  - ngrok을 통한 외부 접근 URL 생성 (수동 실행 시)
  - 최대 1시간 동안 서버 실행

**사용 방법:**

1. GitHub 저장소 > **Actions** 탭
2. **Preview Website** 워크플로우 선택
3. **Run workflow** 클릭
4. 실행 후 **"Setup and start ngrok tunnel"** 단계에서 공개 URL 확인

**필수 GitHub Secrets:**

- `OPENAI_API_KEY`: OpenAI API 키
- `DB_URL`: PostgreSQL 데이터베이스 연결 문자열 (선택사항)
- `NGROK_AUTH_TOKEN`: ngrok 인증 토큰 (외부 접근용, 선택사항)

### Docker 배포

Docker 배포는 향후 지원 예정입니다.

### 프로덕션 설정

- **워커 수**: CPU 코어 수에 맞게 설정
- **리버스 프록시**: Nginx 사용 권장
- **HTTPS**: SSL/TLS 인증서 설정
- **모니터링**: 로그 수집 및 모니터링 도구 연동

## 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### 코딩 스타일

- Python: PEP 8 준수
- 함수/변수명: snake_case 사용
- 클래스명: PascalCase 사용
- 주석: 한국어 또는 영어

## 변경 이력

### v1.1.0 (2026-01)

- GitHub Actions CI/CD 파이프라인 추가
- 환경 변수 기반 데이터베이스 연결 설정
- Render.com PostgreSQL 지원
- 테이블 없을 때 예외 처리 개선
- ngrok을 통한 외부 접근 지원

### v1.0.0 (2026)

- 초기 릴리스
- 법안 검색 기능
- AI 예측 분석
- 리포트 생성
- 챗봇 기능

## 알려진 이슈

- [ ] 대량 데이터 처리 시 성능 최적화 필요
- [ ] 모바일 반응형 디자인 개선 필요
- [ ] 에러 처리 강화 필요

## 로드맵

- [ ] 사용자 인증 및 권한 관리
- [ ] 법안 즐겨찾기 기능
- [ ] 리포트 PDF 다운로드
- [ ] 알림 기능
- [ ] API 문서 자동 생성 (Swagger/OpenAPI)
- [X] CI/CD 파이프라인 구축 (GitHub Actions)
- [ ] 단위 테스트 및 통합 테스트 추가

## 추가 자료

### 관련 문서

- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [PostgreSQL 문서](https://www.postgresql.org/docs/)
- [OpenAI API 문서](https://platform.openai.com/docs/)
- [CatBoost 문서](https://catboost.ai/docs/)

### 유용한 링크

- [법제처 법령정보시스템](https://www.law.go.kr/)
- [국회 법률정보시스템](https://likms.assembly.go.kr/)

## FAQ

### Q: 모델 예측 정확도는 얼마인가요?

A: 모델 성능은 학습 데이터와 평가 지표에 따라 다릅니다. `model/model_final.ipynb`에서 확인할 수 있습니다.

### Q: 데이터베이스에 데이터가 없으면 어떻게 하나요?

A: 먼저 `data/`의 수집·통합 스크립트(`collecting_bill_data_final.py`, `assemble_data_final.py`)를 실행해 원시 데이터를 적재한 뒤, 모델 학습 및 리포트 생성 노트북을 실행하세요. 테이블이 없거나 데이터가 없어도 애플리케이션은 정상적으로 실행되며 빈 결과를 반환합니다.

### Q: GitHub Actions에서 데이터베이스 연결이 안 되면?

A: GitHub Secrets에 `DB_URL`을 설정하세요. Render.com 등의 클라우드 데이터베이스를 사용하는 것을 권장합니다. `DB_URL`이 없으면 테스트용 로컬 PostgreSQL 컨테이너가 사용됩니다 (빈 데이터베이스).

### Q: 챗봇이 답변을 하지 않으면?

A: OpenAI API 키가 올바른지 확인하고, 네트워크 연결을 확인하세요. 리포트 내용이 있는지도 확인해야 합니다.

### Q: 다른 포트로 실행하려면?

A: uvicorn 실행 시 `--port` 옵션을 사용하세요:

```bash
uvicorn main:app --port 8000
```

## 📄 라이선스

이 프로젝트는 CG INSIDE의 소유입니다. 무단 사용 및 배포를 금지합니다.

## 문의

프로젝트 관련 문의사항은 GitHub Issues를 통해 등록해주세요.

---

**© 2026 LawDict AI Legislative Platform | Project by CG INSIDE**
