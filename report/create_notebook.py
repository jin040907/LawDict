import json

# 기본 노트북 구조 생성
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.14"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# 셀 1: 마크다운
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 유사 사례 비교 분석 뉴스 수집\n",
        "\n",
        "이 노트북은 보고서의 \"유사 사례 비교 분석\" 섹션에서 키워드를 추출하고, 딥서치뉴스 API를 통해 관련 뉴스를 검색하여 데이터베이스의 `news` 컬럼에 저장합니다."
    ]
})

# 셀 2: 라이브러리 import
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "# 필요한 라이브러리 import\n",
        "import pandas as pd\n",
        "import requests\n",
        "import json\n",
        "import re\n",
        "from sqlalchemy import create_engine, text, inspect\n",
        "from datetime import datetime, timedelta\n",
        "import time\n",
        "from tqdm import tqdm\n",
        "\n",
        "# 딥서치뉴스 API 설정\n",
        "DEEPSEARCH_API_KEY = \"95b4cee007844c74b5502c57dd41de86\"\n",
        "DEEPSEARCH_API_URL = \"https://news.deepsearch.com/api/\"\n",
        "\n",
        "# 데이터베이스 연결\n",
        "DB_URL = \"postgresql://cginside19:1234@localhost:5432/bill_db\"\n",
        "engine = create_engine(DB_URL)\n",
        "\n",
        "print(\"라이브러리 import 완료\")"
    ],
    "outputs": []
})

# 셀 3: news 컬럼 추가
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "# news 컬럼이 없으면 추가\n",
        "inspector = inspect(engine)\n",
        "columns = [col['name'] for col in inspector.get_columns('final_training_data_copy_sample10_md')]\n",
        "\n",
        "if 'news' not in columns:\n",
        "    print(\"news 컬럼이 없습니다. 추가 중...\")\n",
        "    with engine.connect() as conn:\n",
        "        conn.execute(text(\"ALTER TABLE public.final_training_data_copy_sample10_md ADD COLUMN IF NOT EXISTS news TEXT\"))\n",
        "        conn.commit()\n",
        "    print(\"news 컬럼 추가 완료\")\n",
        "else:\n",
        "    print(\"news 컬럼이 이미 존재합니다.\")"
    ],
    "outputs": []
})

# 셀 4: 유사 사례 섹션 추출 함수
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "def extract_similar_cases_section(report_md: str) -> str:\n",
        "    \"\"\"\n",
        "    report_md에서 유사 사례 비교 분석 섹션(섹션 4)을 추출합니다.\n",
        "    \"\"\"\n",
        "    if not report_md or report_md.strip() == \"\":\n",
        "        return \"\"\n",
        "    \n",
        "    # 섹션 4 추출: ## 4. 유사 사례 비교분석\n",
        "    pattern = r'##\\s+4\\.\\s+유사\\s+사례\\s+비교분석(.*?)(?=##\\s+5\\.|$)'\n",
        "    match = re.search(pattern, report_md, re.DOTALL | re.IGNORECASE)\n",
        "    \n",
        "    if match:\n",
        "        return match.group(1).strip()\n",
        "    return \"\"\n",
        "\n",
        "def extract_keywords_from_similar_cases(section_text: str) -> list:\n",
        "    \"\"\"\n",
        "    유사 사례 비교 분석 섹션에서 키워드를 추출합니다.\n",
        "    - 법안명\n",
        "    - 핵심 차이 내용\n",
        "    - 처리 결과\n",
        "    \"\"\"\n",
        "    keywords = []\n",
        "    \n",
        "    if not section_text:\n",
        "        return keywords\n",
        "    \n",
        "    # 테이블에서 법안명 추출\n",
        "    # | 순위 | 법안명 | 처리 결과 | 유사도 | 핵심 차이 |\n",
        "    table_pattern = r'\\|\\s*\\d+\\s*\\|\\s*([^\\|]+)\\s*\\|\\s*([^\\|]+)\\s*\\|\\s*[\\d.]+\\s*\\|\\s*([^\\|]+)\\s*\\|'\n",
        "    matches = re.finditer(table_pattern, section_text)\n",
        "    \n",
        "    for match in matches:\n",
        "        bill_name = match.group(1).strip()\n",
        "        result = match.group(2).strip()\n",
        "        difference = match.group(3).strip()\n",
        "        \n",
        "        # 법안명에서 주요 키워드 추출 (괄호 제거)\n",
        "        bill_name_clean = re.sub(r'\\([^)]*\\)', '', bill_name).strip()\n",
        "        if bill_name_clean:\n",
        "            keywords.append(bill_name_clean)\n",
        "        \n",
        "        # 핵심 차이에서 키워드 추출 (마크다운 제거)\n",
        "        if difference and difference != \"-\":\n",
        "            difference_clean = re.sub(r'\\*\\*|`|#', '', difference).strip()\n",
        "            if difference_clean and len(difference_clean) > 3:\n",
        "                keywords.append(difference_clean)\n",
        "    \n",
        "    # 중복 제거 및 빈 문자열 제거\n",
        "    keywords = list(set([k for k in keywords if k and len(k) > 2]))\n",
        "    \n",
        "    return keywords[:10]  # 최대 10개 키워드만 사용\n",
        "\n",
        "# 테스트\n",
        "test_text = \"\"\"\n",
        "| 순위 | 법안명 | 처리 결과 | 유사도 | 핵심 차이 |\n",
        "| --- | --- | --- | --- | --- |\n",
        "| 1 | 헌법재판소법 일부개정법률안(정부) | 대안반영폐기 | 0.9210 | 현재 상태: 공포→대안반영폐기 |\n",
        "| 2 | 부동산 실권리자명의 등기에 관한 법률 일부개정법률안(정부) | 원안가결 | 0.9080 | - |\n",
        "\"\"\"\n",
        "keywords = extract_keywords_from_similar_cases(test_text)\n",
        "print(\"추출된 키워드:\", keywords)"
    ],
    "outputs": []
})

# 셀 5: 딥서치뉴스 API 검색 함수
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "def search_news_deepsearch(keywords: list, start_date: str, end_date: str, max_results: int = 10) -> list:\n",
        "    \"\"\"\n",
        "    딥서치뉴스 API를 사용하여 뉴스를 검색합니다.\n",
        "    \n",
        "    Args:\n",
        "        keywords: 검색 키워드 리스트\n",
        "        start_date: 시작 날짜 (YYYY-MM-DD)\n",
        "        end_date: 종료 날짜 (YYYY-MM-DD)\n",
        "        max_results: 최대 결과 개수\n",
        "    \n",
        "    Returns:\n",
        "        뉴스 리스트 (dict 형태)\n",
        "    \"\"\"\n",
        "    if not keywords:\n",
        "        return []\n",
        "    \n",
        "    # 키워드를 하나의 검색어로 결합 (OR 조건)\n",
        "    search_query = \" OR \".join(keywords[:5])  # 최대 5개 키워드만 사용\n",
        "    \n",
        "    try:\n",
        "        # 딥서치뉴스 API 호출\n",
        "        # API 문서에 따라 파라미터 조정 필요\n",
        "        params = {\n",
        "            \"q\": search_query,\n",
        "            \"start_date\": start_date,\n",
        "            \"end_date\": end_date,\n",
        "            \"limit\": max_results\n",
        "        }\n",
        "        \n",
        "        headers = {\n",
        "            \"X-API-Key\": DEEPSEARCH_API_KEY,\n",
        "            \"Content-Type\": \"application/json\"\n",
        "        }\n",
        "        \n",
        "        # GET 요청 시도\n",
        "        response = requests.get(\n",
        "            DEEPSEARCH_API_URL + \"search\",\n",
        "            params=params,\n",
        "            headers=headers,\n",
        "            timeout=30\n",
        "        )\n",
        "        \n",
        "        if response.status_code == 200:\n",
        "            data = response.json()\n",
        "            # API 응답 구조에 따라 파싱 조정 필요\n",
        "            if isinstance(data, dict):\n",
        "                return data.get(\"results\", data.get(\"articles\", data.get(\"news\", [])))\n",
        "            elif isinstance(data, list):\n",
        "                return data\n",
        "            else:\n",
        "                print(f\"예상치 못한 응답 형식: {type(data)}\")\n",
        "                return []\n",
        "        else:\n",
        "            # POST 요청 시도\n",
        "            try:\n",
        "                response = requests.post(\n",
        "                    DEEPSEARCH_API_URL + \"search\",\n",
        "                    json=params,\n",
        "                    headers=headers,\n",
        "                    timeout=30\n",
        "                )\n",
        "                if response.status_code == 200:\n",
        "                    data = response.json()\n",
        "                    if isinstance(data, dict):\n",
        "                        return data.get(\"results\", data.get(\"articles\", data.get(\"news\", [])))\n",
        "                    elif isinstance(data, list):\n",
        "                        return data\n",
        "            except:\n",
        "                pass\n",
        "            \n",
        "            print(f\"API 호출 실패: {response.status_code} - {response.text[:200]}\")\n",
        "            return []\n",
        "            \n",
        "    except Exception as e:\n",
        "        print(f\"뉴스 검색 오류: {e}\")\n",
        "        return []\n",
        "\n",
        "# 테스트 (실제 API 호출 전에 주석 처리)\n",
        "# test_keywords = [\"헌법재판소법\", \"일부개정법률안\"]\n",
        "# test_news = search_news_deepsearch(test_keywords, \"2019-01-01\", \"2019-12-31\", 5)\n",
        "# print(f\"테스트 결과: {len(test_news)}개 뉴스\")"
    ],
    "outputs": []
})

# 셀 6: 날짜 범위 계산 함수
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "def get_date_range_for_bill(propose_dt) -> tuple:\n",
        "    \"\"\"\n",
        "    법안 발의일을 기준으로 뉴스 검색 날짜 범위를 계산합니다.\n",
        "    발의일 전후 6개월 범위로 설정합니다.\n",
        "    \"\"\"\n",
        "    if pd.isna(propose_dt):\n",
        "        # 기본값: 현재 날짜 기준\n",
        "        end_date = datetime.now()\n",
        "        start_date = end_date - timedelta(days=180)\n",
        "    else:\n",
        "        if isinstance(propose_dt, str):\n",
        "            try:\n",
        "                propose_dt = pd.to_datetime(propose_dt)\n",
        "            except:\n",
        "                end_date = datetime.now()\n",
        "                start_date = end_date - timedelta(days=180)\n",
        "                return start_date.strftime(\"%Y-%m-%d\"), end_date.strftime(\"%Y-%m-%d\")\n",
        "        \n",
        "        # 발의일 전후 6개월\n",
        "        start_date = propose_dt - timedelta(days=180)\n",
        "        end_date = propose_dt + timedelta(days=180)\n",
        "    \n",
        "    return start_date.strftime(\"%Y-%m-%d\"), end_date.strftime(\"%Y-%m-%d\")\n",
        "\n",
        "# 테스트\n",
        "test_date = pd.to_datetime(\"2019-10-23\")\n",
        "start, end = get_date_range_for_bill(test_date)\n",
        "print(f\"날짜 범위: {start} ~ {end}\")"
    ],
    "outputs": []
})

# 셀 7: 데이터 로드
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "# 데이터베이스에서 report_md가 있는 모든 레코드 가져오기\n",
        "with engine.connect() as conn:\n",
        "    query = text(\"\"\"\n",
        "        SELECT bill_id, report_md, propose_dt\n",
        "        FROM public.final_training_data_copy_sample10_md\n",
        "        WHERE report_md IS NOT NULL \n",
        "          AND report_md != ''\n",
        "          AND report_md != '준비 중'\n",
        "        ORDER BY propose_dt DESC\n",
        "    \"\"\")\n",
        "    \n",
        "    df = pd.read_sql(query, conn)\n",
        "    \n",
        "print(f\"총 {len(df)}건의 보고서를 처리합니다.\")\n",
        "print(f\"샘플 데이터:\")\n",
        "print(df[['bill_id', 'propose_dt']].head())"
    ],
    "outputs": []
})

# 셀 8: 뉴스 수집 및 저장
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "# 각 레코드에 대해 뉴스 수집 및 저장\n",
        "results = []\n",
        "\n",
        "for idx, row in tqdm(df.iterrows(), total=len(df), desc=\"뉴스 수집 중\"):\n",
        "    bill_id = row['bill_id']\n",
        "    report_md = row['report_md']\n",
        "    propose_dt = row['propose_dt']\n",
        "    \n",
        "    try:\n",
        "        # 1. 유사 사례 비교 분석 섹션 추출\n",
        "        similar_cases_section = extract_similar_cases_section(report_md)\n",
        "        \n",
        "        if not similar_cases_section:\n",
        "            print(f\"[{idx+1}/{len(df)}] {bill_id}: 유사 사례 비교 분석 섹션을 찾을 수 없습니다.\")\n",
        "            results.append({\n",
        "                'bill_id': bill_id,\n",
        "                'news': json.dumps([], ensure_ascii=False),\n",
        "                'status': 'no_section'\n",
        "            })\n",
        "            continue\n",
        "        \n",
        "        # 2. 키워드 추출\n",
        "        keywords = extract_keywords_from_similar_cases(similar_cases_section)\n",
        "        \n",
        "        if not keywords:\n",
        "            print(f\"[{idx+1}/{len(df)}] {bill_id}: 키워드를 추출할 수 없습니다.\")\n",
        "            results.append({\n",
        "                'bill_id': bill_id,\n",
        "                'news': json.dumps([], ensure_ascii=False),\n",
        "                'status': 'no_keywords'\n",
        "            })\n",
        "            continue\n",
        "        \n",
        "        print(f\"[{idx+1}/{len(df)}] {bill_id}: 키워드 {len(keywords)}개 추출 - {keywords[:3]}\")\n",
        "        \n",
        "        # 3. 날짜 범위 계산\n",
        "        start_date, end_date = get_date_range_for_bill(propose_dt)\n",
        "        \n",
        "        # 4. 뉴스 검색\n",
        "        news_list = search_news_deepsearch(keywords, start_date, end_date, max_results=10)\n",
        "        \n",
        "        if not news_list:\n",
        "            print(f\"[{idx+1}/{len(df)}] {bill_id}: 뉴스를 찾을 수 없습니다.\")\n",
        "            results.append({\n",
        "                'bill_id': bill_id,\n",
        "                'news': json.dumps([], ensure_ascii=False),\n",
        "                'status': 'no_news'\n",
        "            })\n",
        "        else:\n",
        "            print(f\"[{idx+1}/{len(df)}] {bill_id}: {len(news_list)}개 뉴스 발견\")\n",
        "            results.append({\n",
        "                'bill_id': bill_id,\n",
        "                'news': json.dumps(news_list, ensure_ascii=False),\n",
        "                'status': 'success'\n",
        "            })\n",
        "        \n",
        "        # API 호출 제한을 위한 딜레이\n",
        "        time.sleep(0.5)\n",
        "        \n",
        "    except Exception as e:\n",
        "        print(f\"[{idx+1}/{len(df)}] {bill_id}: 오류 발생 - {e}\")\n",
        "        results.append({\n",
        "            'bill_id': bill_id,\n",
        "            'news': json.dumps([], ensure_ascii=False),\n",
        "            'status': f'error: {str(e)}'\n",
        "        })\n",
        "\n",
        "print(f\"\\n처리 완료: {len(results)}건\")"
    ],
    "outputs": []
})

# 셀 9: 결과 확인
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "# 결과를 데이터프레임으로 변환\n",
        "results_df = pd.DataFrame(results)\n",
        "print(\"결과 통계:\")\n",
        "print(results_df['status'].value_counts())\n",
        "print(f\"\\n성공: {len(results_df[results_df['status'] == 'success'])}건\")\n",
        "print(f\"실패: {len(results_df[results_df['status'] != 'success'])}건\")"
    ],
    "outputs": []
})

# 셀 10: 데이터베이스 업데이트
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "# 데이터베이스에 news 컬럼 업데이트\n",
        "with engine.connect() as conn:\n",
        "    updated_count = 0\n",
        "    for _, row in results_df.iterrows():\n",
        "        try:\n",
        "            update_query = text(\"\"\"\n",
        "                UPDATE public.final_training_data_copy_sample10_md\n",
        "                SET news = :news\n",
        "                WHERE bill_id = :bill_id\n",
        "            \"\"\")\n",
        "            conn.execute(update_query, {\n",
        "                'news': row['news'],\n",
        "                'bill_id': row['bill_id']\n",
        "            })\n",
        "            updated_count += 1\n",
        "        except Exception as e:\n",
        "            print(f\"업데이트 실패 ({row['bill_id']}): {e}\")\n",
        "    \n",
        "    conn.commit()\n",
        "    print(f\"\\n데이터베이스 업데이트 완료: {updated_count}건\")"
    ],
    "outputs": []
})

# 셀 11: 샘플 결과 확인
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "# 샘플 결과 확인\n",
        "sample_success = results_df[results_df['status'] == 'success'].head(1)\n",
        "if len(sample_success) > 0:\n",
        "    sample_news = json.loads(sample_success.iloc[0]['news'])\n",
        "    print(f\"샘플 뉴스 ({sample_success.iloc[0]['bill_id']}):\")\n",
        "    print(f\"뉴스 개수: {len(sample_news)}\")\n",
        "    if sample_news:\n",
        "        print(f\"첫 번째 뉴스:\")\n",
        "        print(json.dumps(sample_news[0], indent=2, ensure_ascii=False))"
    ],
    "outputs": []
})

with open('fetch_news.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print('노트북 파일 생성 완료')
