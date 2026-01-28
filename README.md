# LawDict

AI 기반 입법 분석 플랫폼

LawDict는 머신러닝과 AI를 활용하여 법안의 처리 결과를 예측하고, 상세한 분석 리포트를 제공하는 웹 애플리케이션입니다.

## 주요 기능

- **법안 검색 및 조회**: 법안명, 발의자명, 발의일을 기준으로 법안 검색
- **AI 예측 분석**: 머신러닝 모델을 통한 법안 처리 결과 예측 및 확률 제공
- **상세 분석 리포트**: 법안 정보, 예측 결과, 근거 분석, 유사 사례 비교 등 종합 리포트 제공
- **AI 챗봇**: 법안 분석 내용에 대한 질문에 실시간으로 답변하는 AI 어시스턴트
- **뉴스 연동**: 유사 법안 관련 뉴스 수집 및 표시

## 프로젝트 구조

```
LawDict/
├── service/              # FastAPI 웹 서버
│   ├── main.py          # 메인 애플리케이션 파일
│   └── templates/       # HTML 템플릿
│       ├── index.html   # 법안 목록 페이지
│       └── detail.html  # 법안 상세 페이지
├── model/               # 머신러닝 모델 관련
│   ├── model_final.ipynb              # 모델 학습/평가 노트북
│   └── thresholds_8class_final.json  # 모델 임계값 설정
└── report/              # 리포트 생성 관련
    ├── ai_report_column.ipynb  # AI 리포트 생성 노트북
    ├── news_column.ipynb       # 뉴스 수집 노트북
    └── create_notebook.py      # 노트북 생성 스크립트
```

## 기술 스택

### Backend
- **FastAPI**: Python 웹 프레임워크
- **PostgreSQL**: 데이터베이스
- **SQLAlchemy**: ORM
- **Pandas**: 데이터 처리
- **OpenAI API**: AI 챗봇 기능
- **DuckDuckGo Search**: 웹 검색 기능

### Frontend
- **Tailwind CSS**: 스타일링
- **Chart.js**: 데이터 시각화
- **Jinja2**: 템플릿 엔진

### Machine Learning
- **Python**: 머신러닝 모델 개발
- **Jupyter Notebook**: 모델 학습 및 분석

## 설치 및 실행

### 사전 요구사항

- Python 3.10 이상
- PostgreSQL 12 이상
- Node.js (선택사항, 개발용)

### 1. 저장소 클론

```bash
git clone <repository-url>
cd LawDict
```

### 2. 가상 환경 생성 및 활성화

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

주요 패키지:
- fastapi
- uvicorn
- sqlalchemy
- pandas
- python-dotenv
- openai
- markdown
- duckduckgo-search
- jinja2

### 4. 환경 변수 설정

`service/.env` 파일을 생성하고 다음 내용을 추가하세요:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. 데이터베이스 설정

PostgreSQL 데이터베이스를 생성하고 `service/main.py`의 DB 연결 정보를 수정하세요:

```python
DB_URL = "postgresql://username:password@localhost:5432/bill_db"
```

필요한 테이블:
- `public.final_training_data_copy_sample10_md`: 법안 데이터 테이블

### 6. 서버 실행

```bash
cd service
python main.py
```

또는 uvicorn을 직접 사용:

```bash
cd service
uvicorn main:app --host 0.0.0.0 --port 8888 --reload
```

서버가 실행되면 브라우저에서 `http://localhost:8888`로 접속할 수 있습니다.

## 사용 방법

### 법안 검색

1. 메인 페이지(`/`)에서 법안 목록을 확인합니다.
2. 검색창에 법안명 또는 발의자명을 입력하여 검색합니다.
3. 발의일 필터(연/월/일)를 사용하여 날짜 범위로 필터링할 수 있습니다.
4. 정렬 옵션을 선택하여 목록을 정렬합니다.

### 법안 상세 분석

1. 법안 목록에서 "AI 리포트 보기" 버튼을 클릭합니다.
2. 상세 페이지에서 다음 정보를 확인할 수 있습니다:
   - **AI 모델 예측 및 분석**: 예측 확률 도넛 차트 및 상세 분석
   - **법안 정보 및 요약**: 법안 기본 정보 및 요약
   - **데이터 기반 근거 분석**: 예측 근거 데이터 분석
   - **유사 법안 비교 분석**: 유사한 법안과의 비교 및 관련 뉴스
   - **사회적 영향력 분석**: 법안의 사회적 영향 분석
   - **향후 대응 방향 및 제언**: 대응 전략 제안

### AI 챗봇 사용

1. 상세 페이지 오른쪽의 챗봇 패널을 사용합니다.
2. 법안 분석 내용에 대해 질문을 입력합니다.
3. AI가 리포트 내용을 바탕으로 답변을 제공합니다.

## 모델 학습

### 모델 학습 실행

```bash
cd model
jupyter notebook model_final.ipynb
```

### 리포트 생성

```bash
cd report
jupyter notebook ai_report_column.ipynb
```

### 뉴스 수집

```bash
cd report
jupyter notebook news_column.ipynb
```

## API 엔드포인트

### 법안 검색 API

```
GET /api/search?q={검색어}&year={연도}&month={월}&day={일}
```

응답 예시:
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

### 챗봇 API

```
POST /api/chat
Content-Type: application/json

{
  "message": "질문 내용",
  "bill_name": "법안명",
  "context": "리포트 본문 텍스트"
}
```

응답 예시:
```json
{
  "answer": "답변 내용"
}
```

## 개발 참고사항

### 데이터베이스 스키마

주요 테이블: `public.final_training_data_copy_sample10_md`

주요 컬럼:
- `bill_id`: 법안 ID
- `bill_name`: 법안명
- `proposer_name`: 발의자명
- `propose_dt`: 발의일
- `ai_prediction`: AI 예측 결과
- `ai_probability`: AI 예측 확률
- `ai_report`: AI 요약 리포트
- `report_md`: 상세 분석 리포트 (마크다운)
- `news`: 관련 뉴스 데이터 (JSON)

### 리포트 구조

리포트는 다음 섹션으로 구성됩니다:
- 0. 요약 (summary)
- 1. 입력 법안 정보 (bill_info)
- 2. 모델 예측 결과 (prediction)
- 3. 데이터 기반 근거 분석 (evidence)
- 4. 유사 사례 비교 분석 (similar_cases)
- 5. 사회적 영향력 분석 (social_impact)
- 6. 향후 대응 방향 및 제언 (next_action)

## 라이선스

이 프로젝트는 CG INSIDE에서 개발되었습니다.

## 문의

프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.

---

© 2026 LawDict AI Legislative Platform | Project by CG INSIDE

