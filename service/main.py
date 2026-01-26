from __future__ import annotations

from fastapi import Request

import openai # 혹은 사용 중인 LLM 라이브러리

import re

import json

import markdown

from fastapi import FastAPI, Request, Query

from fastapi.templating import Jinja2Templates

from fastapi.responses import HTMLResponse, JSONResponse

import pandas as pd

from sqlalchemy import create_engine, text

import uvicorn

from duckduckgo_search import DDGS



app = FastAPI()

import os
from dotenv import load_dotenv

import openai

# Load environment variables from .env file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

openai.api_key = os.getenv("OPENAI_API_KEY", "")

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# DB 연결 설정

DB_URL = "postgresql://cginside19:1234@localhost:5432/bill_db"

engine = create_engine(DB_URL)



def remove_report_titles(text: str) -> str:

    """

    보고서에 자동 추가된 제목들을 제거합니다.

    예: "**📋 주요 메타 데이터**", "**📊 예측 확률 분포**" 등

    """

    # 제목 패턴 제거 (이모지 + 제목 형식)

    patterns = [

        r'\*\*[📋📊🔍📈💡]\s*[^\*]+\*\*\s*\n',  # **📋 제목** 형식

        r'\*\*[^\*]+\*\*\s*\n',  # **제목** 형식 (이모지 없는 경우)

    ]

    result = text

    for pattern in patterns:

        result = re.sub(pattern, '', result, flags=re.MULTILINE)

    return result.strip()



def parse_prediction_table(md_text: str) -> list:

    """

    예측 확률 테이블을 파싱합니다.

    마크다운 테이블 형식에서 순위, 예측, 확률을 추출합니다.

    """

    predictions = []

   

    if not md_text:

        return predictions

   

    # "## 2. 모델 예측 결과" 섹션에서 테이블 찾기

    section_2_pattern = r'##\s+2\.\s+모델\s+예측\s+결과(.*?)(?=##\s+3\.|$)'

    section_2_match = re.search(section_2_pattern, md_text, re.DOTALL | re.IGNORECASE)

   

    if not section_2_match:

        return predictions

   

    section_2_content = section_2_match.group(1)

   

    # 테이블 헤더 찾기 (순위, 예측, 확률)

    table_header_pattern = r'\|\s*순위\s*\|\s*예측\s*\|\s*확률\s*\|'

    if not re.search(table_header_pattern, section_2_content):

        return predictions

   

    # 테이블 행 찾기 (| 숫자 | 텍스트 | 숫자.숫자 | 형식)

    table_row_pattern = r'\|\s*(\d+)\s*\|\s*([^\|]+)\s*\|\s*([\d.]+)\s*\|'

    matches = re.finditer(table_row_pattern, section_2_content)

   

    for match in matches:

        try:

            rank = int(match.group(1))

            prediction = match.group(2).strip()

            probability = float(match.group(3))

            predictions.append({

                'rank': rank,

                'prediction': prediction,

                'probability': probability

            })

        except (ValueError, IndexError):

            continue

   

    # 순위 순으로 정렬 (이미 순위대로 나올 가능성이 높지만 확실하게)

    predictions.sort(key=lambda x: x['rank'])

    # Top 3만 반환

    return predictions[:3]



def split_expert_report(md_text: str) -> dict:

    """

    전문가 보고서를 파싱합니다. JSON 형식 또는 마크다운 형식을 자동 감지하여 처리합니다.

    원본 내용을 그대로 가져와서 마크다운→HTML 변환만 수행합니다.

    """

    keys = ["summary", "bill_info", "prediction", "evidence", "similar_cases", "social_impact", "next_action"]

   

    if not md_text or "준비 중" in md_text:

        return {k: "" for k in keys}



    txt = md_text.strip()

   

    # JSON 형식인지 확인 (첫 문자가 {)

    if txt.startswith('{'):

        try:

            data = json.loads(txt)

            result = {}

            # JSON의 원본 내용을 가져오되, 제목 제거 함수 적용

            for k in keys:

                if k in data and data[k]:

                    val = str(data[k]).strip()

                    # 보고서에 자동 추가된 제목 제거 (주요 메타 데이터, 예측 확률 분포 등)

                    val = remove_report_titles(val)

                    md = markdown.Markdown(extensions=['extra', 'nl2br', 'sane_lists', 'tables'])

                    result[k] = md.convert(val)

                else:

                    result[k] = ""

            return result

        except json.JSONDecodeError:

            # JSON 파싱 실패 시 마크다운으로 처리

            pass



    # 마크다운 형식 처리

    # ## 0. ~ ## 6. 형식의 섹션을 찾아서 분리

    section_pattern = r'##\s+(\d+)\.\s+([^\n]+)'

   

    # 모든 섹션 헤더 찾기

    matches = list(re.finditer(section_pattern, txt))

   

    result = {}

   

    for idx, match in enumerate(matches):

        section_num = int(match.group(1))
        
        # 섹션 번호를 키 인덱스로 매핑
        # 0->summary, 1->bill_info, 2->prediction, 3->evidence, 4->similar_cases, 5->social_impact, 6->next_action
        if section_num >= len(keys):
            continue  # 범위를 벗어난 섹션은 건너뜀
        
        key_index = section_num

        # 현재 섹션의 시작 위치 (헤더 다음 줄부터)
        start_pos = match.end()

        # 다음 섹션의 시작 위치 (없으면 끝까지)
        if idx + 1 < len(matches):
            end_pos = matches[idx + 1].start()
        else:
            end_pos = len(txt)

        # 섹션 내용 추출 (헤더 제목은 제외하고 내용만)
        section_body = txt[start_pos:end_pos].strip()
        
        # 디버깅: 섹션 5와 6 파싱 확인
        if section_num == 5 or section_num == 6:
            print(f"[DEBUG] 섹션 {section_num} 파싱 중...")
            print(f"  - 키 인덱스: {key_index}")
            print(f"  - 키 이름: {keys[key_index]}")
            print(f"  - 내용 길이: {len(section_body)}")
            print(f"  - 내용 일부 (처음 200자): {section_body[:200]}")

        # HTML로 변환
        if section_body:
            try:
                # 테이블이 리스트나 다른 블록 바로 다음에 오면 테이블 앞에 빈 줄 추가
                # 마크다운 테이블은 빈 줄로 구분되어야 함
                lines = section_body.split('\n')
                processed_lines = []

                for i, line in enumerate(lines):
                    line_stripped = line.strip()

                    # 테이블 헤더 패턴 발견 (| 컬럼1 | 컬럼2 | 형식, --- 구분선이 아닌 경우)
                    if re.match(r'^\|\s*.+\s*\|', line_stripped) and not re.match(r'^\|\s*[-:]+', line_stripped):
                        # 이전 줄 확인
                        if i > 0:
                            prev_line = lines[i-1].strip()
                            # 이전 줄이 비어있지 않고 테이블이 아니면 빈 줄 추가
                            if prev_line and not prev_line.startswith('|'):
                                # processed_lines의 마지막 항목이 빈 줄이 아니면 추가
                                if not processed_lines or processed_lines[-1].strip():
                                    processed_lines.append('')  # 빈 줄 추가
                                    print(f"[DEBUG] 테이블 헤더 앞에 빈 줄 추가 (라인 {i+1})")

                    processed_lines.append(line)

                section_body_processed = '\n'.join(processed_lines)

                # 마크다운 확장 명시적으로 로드
                md = markdown.Markdown(extensions=['extra', 'nl2br', 'sane_lists', 'tables'])
                html_content = md.convert(section_body_processed)

                # 디버깅: 섹션 4 (similar_cases)의 경우 테이블 포함 여부 확인
                if section_num == 4:
                    # 테이블이 있는지 확인
                    has_table_marker = '|' in section_body
                    has_table_html = '<table' in html_content
                    
                    # 테이블 앞 빈 줄 확인
                    lines_before_table = []
                    for i, line in enumerate(processed_lines):
                        if re.match(r'^\|\s*.+\s*\|', line.strip()) and not re.match(r'^\|\s*[-:]+', line.strip()):
                            # 테이블 헤더 발견
                            if i > 0:
                                prev_line = processed_lines[i-1].strip() if i > 0 else ""
                                lines_before_table.append(f"라인 {i}: 이전 줄 = '{prev_line}' (빈 줄 여부: {not prev_line})")
                            break
                    
                    print(f"[DEBUG] 섹션 4 (유사 사례) 테이블 파싱 상태:")
                    print(f"  - 마크다운에 테이블 마커(|) 있음: {has_table_marker}")
                    print(f"  - HTML에 <table> 태그 있음: {has_table_html}")
                    print(f"  - 테이블 앞 빈 줄 정보: {lines_before_table}")
                    
                    if has_table_marker and not has_table_html:
                        print(f"[WARNING] 섹션 4에 테이블이 HTML로 변환되지 않음!")
                        print(f"[DEBUG] 원본 내용 일부 (처음 500자):\n{section_body[:500]}")
                        print(f"[DEBUG] 처리된 내용 일부 (처음 500자):\n{section_body_processed[:500]}")
                        print(f"[DEBUG] HTML 일부 (처음 500자):\n{html_content[:500]}")
                    elif has_table_html:
                        print(f"[DEBUG] 섹션 4 테이블 변환 성공!")

                result[keys[key_index]] = html_content

            except Exception as e:
                print(f"[ERROR] 마크다운 변환 오류 (섹션 {section_num}): {str(e)}")
                import traceback
                traceback.print_exc()
                result[keys[key_index]] = ""

        else:
            result[keys[key_index]] = ""

   

    # 없는 키는 빈 문자열로 채움

    for k in keys:

        if k not in result:

            result[k] = ""

           

    return result



@app.get("/api/search")
async def search_bills(
    request: Request, 
    q: str = Query("", description="검색어"),
    year: int = Query(None, description="연도"),
    month: int = Query(None, description="월"),
    day: int = Query(None, description="일")
):
    """
    법안 검색 API - 전체 데이터에서 검색
    """
    import sys
    print(f"[DEBUG] ========== Search API called ==========", file=sys.stderr, flush=True)
    print(f"[DEBUG] q='{q}' (type: {type(q).__name__}, len: {len(q) if q else 0})", file=sys.stderr, flush=True)
    print(f"[DEBUG] year={year} (type: {type(year).__name__ if year is not None else 'NoneType'})", file=sys.stderr, flush=True)
    print(f"[DEBUG] month={month} (type: {type(month).__name__ if month is not None else 'NoneType'})", file=sys.stderr, flush=True)
    print(f"[DEBUG] day={day} (type: {type(day).__name__ if day is not None else 'NoneType'})", file=sys.stderr, flush=True)
    print(f"[DEBUG] Request URL: {request.url}", file=sys.stderr, flush=True)
    print(f"[DEBUG] Query params dict: {dict(request.query_params)}", file=sys.stderr, flush=True)
    
    try:
        # 데이터베이스 연결 확인
        print(f"[DEBUG] Engine URL: {engine.url}", file=sys.stderr, flush=True)
        print(f"[DEBUG] Testing database connection...", file=sys.stderr, flush=True)
        
        with engine.connect() as conn:
            print(f"[DEBUG] Database connection successful", file=sys.stderr, flush=True)
            # 기본 쿼리 구성
            where_conditions = []
            params = {}
            
            # 검색어 조건 - PostgreSQL의 ILIKE 사용 (대소문자 구분 없음)
            if q and q.strip():
                search_term = f"%{q.strip()}%"
                # ILIKE는 PostgreSQL에서 대소문자를 구분하지 않는 LIKE
                where_conditions.append("(bill_name ILIKE :search_term OR proposer_name ILIKE :search_term)")
                params["search_term"] = search_term
                print(f"[DEBUG] Search term added: {search_term}", file=sys.stderr, flush=True)
                print(f"[DEBUG] Search term type: {type(search_term)}, encoding: {search_term.encode('utf-8')}", file=sys.stderr, flush=True)
            else:
                print(f"[DEBUG] No search term provided or empty", file=sys.stderr, flush=True)
            
            # 발의일 필터 조건
            if year is not None:
                where_conditions.append("EXTRACT(YEAR FROM propose_dt) = :year")
                params["year"] = int(year)
                print(f"[DEBUG] Year filter: {year}", file=sys.stderr, flush=True)
            if month is not None:
                where_conditions.append("EXTRACT(MONTH FROM propose_dt) = :month")
                params["month"] = int(month)
                print(f"[DEBUG] Month filter: {month}", file=sys.stderr, flush=True)
            if day is not None:
                where_conditions.append("EXTRACT(DAY FROM propose_dt) = :day")
                params["day"] = int(day)
                print(f"[DEBUG] Day filter: {day}", file=sys.stderr, flush=True)
            
            # WHERE 절 구성
            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)
            
            print(f"[DEBUG] WHERE clause: {where_clause}", file=sys.stderr, flush=True)
            print(f"[DEBUG] Params: {params}", file=sys.stderr, flush=True)
            
            # 검색 쿼리 실행 - 더 안전한 방식으로 수정
            base_query = """
                SELECT bill_id, bill_name, proposer_name, propose_dt
                FROM public.final_training_data_copy_sample10_md
            """
            
            if where_clause:
                query = text(base_query + " " + where_clause + " ORDER BY propose_dt DESC, bill_id DESC")
            else:
                query = text(base_query + " ORDER BY propose_dt DESC, bill_id DESC")
            
            results = []
            try:
                print(f"[DEBUG] Executing query...", file=sys.stderr, flush=True)
                print(f"[DEBUG] Query text: {str(query)}", file=sys.stderr, flush=True)
                print(f"[DEBUG] Query params: {params}", file=sys.stderr, flush=True)
                
                df_results = pd.read_sql(query, conn, params=params)
                print(f"[DEBUG] DataFrame shape: {df_results.shape}", file=sys.stderr, flush=True)
                print(f"[DEBUG] DataFrame columns: {df_results.columns.tolist()}", file=sys.stderr, flush=True)
                
                results = df_results.to_dict(orient="records")
                print(f"[DEBUG] Search results count: {len(results)}", file=sys.stderr, flush=True)
                if len(results) > 0:
                    print(f"[DEBUG] First result keys: {list(results[0].keys())}", file=sys.stderr, flush=True)
                    print(f"[DEBUG] First result sample: bill_id={results[0].get('bill_id')}, bill_name={results[0].get('bill_name')[:50] if results[0].get('bill_name') else None}", file=sys.stderr, flush=True)
                else:
                    print(f"[DEBUG] No results found - checking if query is correct...", file=sys.stderr, flush=True)
                    # 간단한 테스트 쿼리 실행
                    test_query = text("SELECT COUNT(*) as cnt FROM public.final_training_data_copy_sample10_md")
                    test_result = pd.read_sql(test_query, conn)
                    total_rows = test_result.iloc[0]['cnt']
                    print(f"[DEBUG] Total rows in table: {total_rows}", file=sys.stderr, flush=True)
                    
                    # 검색어로 직접 테스트
                    if q and q.strip():
                        test_search_query = text("""
                            SELECT COUNT(*) as cnt 
                            FROM public.final_training_data_copy_sample10_md
                            WHERE bill_name ILIKE :search_term OR proposer_name ILIKE :search_term
                        """)
                        test_search_result = pd.read_sql(test_search_query, conn, params={'search_term': f'%{q.strip()}%'})
                        search_count = test_search_result.iloc[0]['cnt']
                        print(f"[DEBUG] Direct search test for '{q}': {search_count} matches found", file=sys.stderr, flush=True)
                        
                        if search_count > 0:
                            # 실제 매칭되는 데이터 샘플 확인
                            sample_query = text("""
                                SELECT bill_id, bill_name, proposer_name 
                                FROM public.final_training_data_copy_sample10_md
                                WHERE bill_name ILIKE :search_term OR proposer_name ILIKE :search_term
                                LIMIT 3
                            """)
                            sample_result = pd.read_sql(sample_query, conn, params={'search_term': f'%{q.strip()}%'})
                            print(f"[DEBUG] Sample matching records:", file=sys.stderr, flush=True)
                            for idx, row in sample_result.iterrows():
                                print(f"  - bill_id: {row['bill_id']}, bill_name: {row['bill_name'][:50]}, proposer: {row['proposer_name']}", file=sys.stderr, flush=True)
            except Exception as db_error:
                print(f"[ERROR] Database query error: {db_error}", file=sys.stderr, flush=True)
                import traceback
                traceback.print_exc(file=sys.stderr)
                results = []
            
            # 날짜 포맷팅 및 JSON 직렬화 가능하도록 변환
            print(f"[DEBUG] Formatting dates for {len(results)} results...", file=sys.stderr, flush=True)
            for b in results:
                pd_val = b.get("propose_dt")
                if pd_val is not None:
                    try:
                        # Timestamp를 문자열로 변환
                        if hasattr(pd_val, "strftime"):
                            date_str = pd_val.strftime("%Y-%m-%d")
                        elif hasattr(pd_val, "isoformat"):
                            date_str = pd_val.isoformat()[:10]
                        else:
                            str_val = str(pd_val)
                            if len(str_val) >= 10:
                                date_str = str_val[:10]
                            else:
                                date_str = ""
                        
                        b["propose_dt_str"] = date_str
                        # 원본 Timestamp 객체를 문자열로 교체 (JSON 직렬화를 위해)
                        b["propose_dt"] = date_str
                    except Exception as e:
                        print(f"Error formatting propose_dt: {e}", file=sys.stderr, flush=True)
                        b["propose_dt_str"] = ""
                        b["propose_dt"] = ""
                else:
                    b["propose_dt_str"] = ""
                    b["propose_dt"] = ""
            
            response_data = {"results": results, "count": len(results)}
            print(f"[DEBUG] Returning response: count={len(results)}", file=sys.stderr, flush=True)
            print(f"[DEBUG] Response data sample: {response_data if len(results) == 0 else {'count': len(results), 'first_result_keys': list(results[0].keys()) if results else []}}", file=sys.stderr, flush=True)
            return JSONResponse(response_data)
    
    except Exception as e:
        print(f"[ERROR] Search error: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return JSONResponse({"results": [], "count": 0})


@app.get("/")

async def main_page(request: Request, page: int = 1):

    limit = 5

    offset = (page - 1) * limit

   

    with engine.connect() as conn:

        # 전체 개수 파악

        total_count = conn.execute(text("SELECT COUNT(*) FROM public.final_training_data_copy_sample10_md")).scalar()

        total_pages = (total_count // limit) + (1 if total_count % limit > 0 else 0)

        # 발의일 필터 옵션: 연 2008~2026, 월 1~12, 일 1~31
        available_years = list(range(2026, 2007, -1))  # 2026, 2025, ..., 2008
        available_months = list(range(1, 13))          # 1~12
        available_days = list(range(1, 32))            # 1~31

        # 목록 조회 (SQL 인젝션 방지를 위해 text() 사용)

        query = text("""

            SELECT bill_id, bill_name, proposer_name, propose_dt

            FROM public.final_training_data_copy_sample10_md

            ORDER BY propose_dt DESC, bill_id DESC

            LIMIT :limit OFFSET :offset

        """)

        latest_bills = pd.read_sql(query, conn, params={"limit": limit, "offset": offset}).to_dict(orient="records")

        for b in latest_bills:
            pd_val = b.get("propose_dt")
            if pd_val is not None:
                try:
                    if hasattr(pd_val, "strftime"):
                        b["propose_dt_str"] = pd_val.strftime("%Y-%m-%d")
                    elif hasattr(pd_val, "isoformat"):
                        b["propose_dt_str"] = pd_val.isoformat()[:10]
                    else:
                        str_val = str(pd_val)
                        if len(str_val) >= 10:
                            b["propose_dt_str"] = str_val[:10]
                        else:
                            b["propose_dt_str"] = ""
                except Exception as e:
                    print(f"Error formatting propose_dt: {e}, value: {pd_val}, type: {type(pd_val)}")
                    b["propose_dt_str"] = ""
            else:
                b["propose_dt_str"] = ""

   

    return templates.TemplateResponse("index.html", {

        "request": request,

        "latest_bills": latest_bills,

        "current_page": page,

        "total_pages": total_pages,

        "available_years": available_years,

        "available_months": available_months,

        "available_days": available_days,

    })



@app.get("/bill/{bill_id}")

async def bill_detail(request: Request, bill_id: str):

    query = text("""

        SELECT bill_name, proposer_name, ai_prediction, ai_probability, ai_report, report_md, news

        FROM public.final_training_data_copy_sample10_md

        WHERE bill_id = :bill_id

    """)

   

    # 뉴스 데이터 초기화 (예외 처리 전에 미리 정의)
    news_data = []
    
    try:

        with engine.connect() as conn:

            bill_df = pd.read_sql(query, conn, params={"bill_id": bill_id})

           

        if bill_df.empty:

            return HTMLResponse("<h1>DB에 해당 ID가 없습니다.</h1>", status_code=404)

       

        data = bill_df.iloc[0]

       

        # 1. 보고서 파싱 (섹션별로 쪼개기) - report_md 컬럼에서 가져오기

        raw_report_md = data['report_md'] if data['report_md'] else "전문가 분석 보고서가 준비 중입니다."
        
        # 디버깅: report_md 내용 확인
        print(f"[DEBUG] bill_id: {bill_id}")
        print(f"[DEBUG] report_md 타입: {type(raw_report_md)}")
        print(f"[DEBUG] report_md 길이: {len(str(raw_report_md)) if raw_report_md else 0}")
        if raw_report_md:
            # 섹션 5와 6이 있는지 확인
            has_section5 = "## 5." in str(raw_report_md) or "##5." in str(raw_report_md)
            has_section6 = "## 6." in str(raw_report_md) or "##6." in str(raw_report_md)
            print(f"[DEBUG] 섹션 5 포함 여부: {has_section5}")
            print(f"[DEBUG] 섹션 6 포함 여부: {has_section6}")
            if has_section5:
                # 섹션 5 내용 일부 확인
                section5_match = re.search(r'##\s*5\.\s*[^\n]+\n(.*?)(?=##\s*6\.|$)', str(raw_report_md), re.DOTALL)
                if section5_match:
                    section5_content = section5_match.group(1).strip()[:200]
                    print(f"[DEBUG] 섹션 5 내용 일부: {section5_content}")

        parsed_sections = split_expert_report(raw_report_md)

       

        # 1-1. 예측 확률 테이블 파싱 (도넛 그래프용)

        prediction_data = parse_prediction_table(raw_report_md)
        
        # 1-2. 뉴스 데이터 파싱 (유사 사례 비교 분석 섹션용)
        news_data = []
        news_json = data.get('news')
        if news_json:
            try:
                news_list = json.loads(news_json) if isinstance(news_json, str) else news_json
                if isinstance(news_list, list) and len(news_list) > 0:
                    # 최대 10개까지만 사용
                    news_data = news_list[:10]
            except (json.JSONDecodeError, TypeError) as e:
                print(f"[DEBUG] 뉴스 데이터 파싱 오류: {e}")
                news_data = []



        # 2. ai_report를 HTML로 변환

        ai_report_html = ""

        ai_report_raw = data.get('ai_report')

        print(f"[DEBUG] ai_report_raw 타입: {type(ai_report_raw)}, 값: {str(ai_report_raw)[:100] if ai_report_raw else 'None'}")

       

        if ai_report_raw is not None and str(ai_report_raw).strip():

            ai_report_text = str(ai_report_raw).strip()

            print(f"[DEBUG] ai_report 길이: {len(ai_report_text)}")

            try:

                md = markdown.Markdown(extensions=['extra', 'nl2br', 'sane_lists', 'tables'])

                ai_report_html = md.convert(ai_report_text)

               

                # 볼드 태그 제거 (strong, b 태그를 일반 텍스트로)
                # <strong>태그와 <b>태그 제거 (내용은 유지)

                ai_report_html = re.sub(r'</?strong[^>]*>', '', ai_report_html, flags=re.IGNORECASE)

                ai_report_html = re.sub(r'</?b[^>]*>', '', ai_report_html, flags=re.IGNORECASE)

               

                # 1. 맨 윗줄에 법안명 볼드체로 추가 (인라인 style로 확실히 적용)
                bill_name = str(data['bill_name'])
                from markupsafe import escape
                bill_name_safe = escape(bill_name)
                ai_report_html = f'<p><strong class="bill-name-bold" style="font-weight: 700 !important;">{bill_name_safe}</strong></p>\n' + ai_report_html

               

                # 2. "제안 이유:", "주요 내용:", "시사점:" 같은 섹션 제목을 볼드체로 강조
                # 역순으로 처리하여 나중 섹션이 먼저 섹션에 포함되지 않도록 함

                section_titles = ['시사점:', '주요 내용:', '제안 이유:']  # 역순

                for title in section_titles:

                    # 이미 strong.section-title-bold로 감싸져 있으면 스킵

                    if f'<strong class="section-title-bold">{title}</strong>' in ai_report_html:

                        print(f"[DEBUG] '{title}' 이미 처리됨 - 스킵")

                        continue

                   

                    # "주요 내용:"과 "시사점:" 앞에 개행 추가할지 결정

                    needs_break_before = title in ['주요 내용:', '시사점:']

                    break_before = '<br><br>' if needs_break_before else ''

                   

                    # 가장 간단한 패턴: "제안 이유:" 다음에 공백이 있거나 없거나, 다른 섹션 제목이나 태그 끝까지

                    # 공백 있는 경우: "제안 이유: 내용"

                    pattern1 = re.escape(title) + r'(\s+)([^<]+?)(?=주요\s*내용:|시사점:|제안\s*이유:|<|$)'

                    def replace1(m):

                        space = m.group(1)

                        content = m.group(2)

                        return f'{break_before}<strong class="section-title-bold">{title}</strong>{space}{content}'

                    new_html = re.sub(pattern1, replace1, ai_report_html, flags=re.IGNORECASE)

                    if new_html != ai_report_html:

                        ai_report_html = new_html

                        print(f"[DEBUG] '{title}' 볼드 처리 완료 (공백 있음)")

                        continue

                   

                    # 공백 없는 경우: "제안 이유:내용"

                    pattern2 = re.escape(title) + r'([^<\s][^<]+?)(?=주요\s*내용:|시사점:|제안\s*이유:|<|$)'

                    def replace2(m):

                        content = m.group(1)

                        return f'{break_before}<strong class="section-title-bold">{title}</strong> {content}'

                    new_html = re.sub(pattern2, replace2, ai_report_html, flags=re.IGNORECASE)

                    if new_html != ai_report_html:

                        ai_report_html = new_html

                        print(f"[DEBUG] '{title}' 볼드 처리 완료 (공백 없음)")

                    else:

                        print(f"[DEBUG] '{title}' 볼드 처리 실패 - 패턴 매칭 실패")

               

                # 3. 추가 개행 처리: 이미 볼드 처리된 섹션 사이에 개행이 없으면 추가

                # "제안 이유:" 다음에 "주요 내용:"이 같은 <p> 안에 있는 경우

                ai_report_html = re.sub(r'(<strong[^>]*>제안\s*이유:</strong>[^<]+?)(<strong[^>]*>주요\s*내용:</strong>)', r'\1<br><br>\2', ai_report_html, flags=re.IGNORECASE)

                # "주요 내용:" 다음에 "시사점:"이 같은 <p> 안에 있는 경우

                ai_report_html = re.sub(r'(<strong[^>]*>주요\s*내용:</strong>[^<]+?)(<strong[^>]*>시사점:</strong>)', r'\1<br><br>\2', ai_report_html, flags=re.IGNORECASE)

               

                # 중복된 <br><br> 제거 (3개 이상 연속된 경우 2개로)

                ai_report_html = re.sub(r'(<br>\s*){3,}', '<br><br>', ai_report_html, flags=re.IGNORECASE)

               

                # "핵심 요약 리포트:" 부분만 강조 처리
                # 가장 확실한 방법: 첫 번째 <p> 태그에서 "핵심 요약 리포트:" 찾아서 치환

                # 패턴 1: <p>태그 안에 있는 경우 (가장 일반적)

                pattern1 = r'(<p[^>]*>)(핵심\s*요약\s*리포트\s*:)\s*([^<]+?)(</p>)'

                match1 = re.search(pattern1, ai_report_html, flags=re.IGNORECASE)

               

                if match1:

                    p_open = match1.group(1)

                    title = match1.group(2)

                    rest = match1.group(3)

                    p_close = match1.group(4)

                    replacement = f'{p_open}<span class="report-title-highlight">{title}</span>{rest}{p_close}'

                    ai_report_html = ai_report_html.replace(match1.group(0), replacement, 1)

                    print(f"[DEBUG] '핵심 요약 리포트:' 강조 처리 완료 (패턴1)")

                else:

                    # 패턴 2: 첫 번째 <p> 태그 전체를 찾아서 치환

                    first_p_pattern = r'<p[^>]*>.*?핵심\s*요약\s*리포트\s*:.*?</p>'

                    match2 = re.search(first_p_pattern, ai_report_html, flags=re.IGNORECASE | re.DOTALL)

                   

                    if match2:

                        original_p = match2.group(0)

                        # "핵심 요약 리포트:" 부분만 span으로 감싸기

                        replaced_p = original_p.replace('핵심 요약 리포트:', '<span class="report-title-highlight">핵심 요약 리포트:</span>', 1)

                        ai_report_html = ai_report_html.replace(original_p, replaced_p, 1)

                        print(f"[DEBUG] '핵심 요약 리포트:' 강조 처리 완료 (패턴2)")

                    else:

                        # 패턴 3: 단순 문자열 치환 (최후의 수단)

                        if '핵심 요약 리포트:' in ai_report_html:

                            ai_report_html = ai_report_html.replace('핵심 요약 리포트:', '<span class="report-title-highlight">핵심 요약 리포트:</span>', 1)

                            print(f"[DEBUG] '핵심 요약 리포트:' 강조 처리 완료 (패턴3 - 단순 치환)")

                        else:

                            print(f"[DEBUG] '핵심 요약 리포트:' 강조 처리 실패")

                            print(f"[DEBUG] HTML 일부 (처음 500자): {ai_report_html[:500]}")

                            print(f"[DEBUG] '핵심' 포함 여부: {'핵심' in ai_report_html}")

                            print(f"[DEBUG] '요약' 포함 여부: {'요약' in ai_report_html}")

                            if '핵심' in ai_report_html:

                                # 부분 매칭으로 찾기

                                핵심_idx = ai_report_html.find('핵심')

                                print(f"[DEBUG] '핵심' 위치 주변 텍스트: {ai_report_html[max(0, 핵심_idx-50):핵심_idx+100]}")

               

                print(f"[DEBUG] ai_report_html 길이: {len(ai_report_html)}")

            except Exception as e:

                print(f"[ERROR] ai_report 마크다운 변환 오류: {str(e)}")

                ai_report_html = f"<p>{ai_report_text}</p>"  # 마크다운 변환 실패 시 일반 텍스트로

        else:

            print(f"[DEBUG] ai_report가 비어있거나 None입니다.")

       

        # 3. 결과물 조합

        context = {

            "request": request,

            "bill_id": bill_id,

            "bill_name": str(data['bill_name']),

            "proposer_name": str(data['proposer_name']),

            "probability": str(data['ai_probability']) if data['ai_probability'] is not None else "0.0",

            "prediction": str(data['ai_prediction']) if data['ai_prediction'] else "데이터 없음",

            "report": str(data['ai_report']) if data['ai_report'] else "",

            "ai_report_html": ai_report_html,  # HTML로 변환된 ai_report

            "sections": parsed_sections,  # 👈 쪼개진 HTML 섹션들이 담긴 딕셔너리
            "news_data": news_data,  # 👈 뉴스 데이터 리스트 (최대 10개)

            "prediction_data": json.dumps(prediction_data, ensure_ascii=False)  # 👈 도넛 그래프용 예측 확률 데이터 (JSON 문자열)

        }

       

        # 디버깅: 파싱 결과 확인

        print(f"[DEBUG] bill_id: {bill_id}")

        print(f"[DEBUG] report_md 길이: {len(raw_report_md) if raw_report_md else 0}")

        print(f"[DEBUG] sections keys: {list(parsed_sections.keys())}")

        for k in ['summary', 'bill_info', 'prediction', 'evidence', 'similar_cases', 'social_impact', 'next_action']:

            v = parsed_sections.get(k, '')

            has_content = bool(v and v.strip())

            print(f"[DEBUG] {k}: length={len(v)}, has_content={has_content}")

            if not has_content:

                print(f"[DEBUG] ⚠️ {k} is EMPTY!")



        return templates.TemplateResponse("detail.html", context)



    except Exception as e:

        print(f"❌ 에러 발생: {str(e)}")

        return HTMLResponse(content=f"<h1>서버 오류</h1><p>{str(e)}</p>", status_code=500)


def search_web(query: str, max_results: int = 5) -> list:

    print(f"[DEBUG] 실제 검색 쿼리: {query}")  # 추가

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            print(f"[DEBUG] 검색 결과 개수: {len(results)}")  # 추가
            return results
        
    except Exception as e:
        print(f"웹 검색 오류 발생: {e}")
        return []
    
def search_web(query: str, max_results: int = 5) -> list:
    print(f"[DEBUG] 실제 검색 쿼리: {query}")  # 추가
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            print(f"[DEBUG] 검색 결과 개수: {len(results)}")  # 추가
            return results
    except Exception as e:
        print(f"웹 검색 오류 발생: {e}")
        return []
    

@app.post("/api/chat")
async def chat_with_lawdict(request: Request):
    try:
        # 1. 데이터 파싱
        req_json = await request.json()
        user_message = req_json.get("message", "").strip()
        bill_name = req_json.get("bill_name", "").strip()
        report_context = req_json.get("context", "")

        # 2. 실시간 데이터베이스 조회 (최소한의 정보만 빠르게 조회)
        prob_value = 0.0
        pred_value = "데이터 없음"
        
        try:
            with engine.connect() as conn:
                query = text("SELECT ai_prediction, ai_probability FROM public.final_training_data_copy_sample10_md WHERE bill_name = :bill_name LIMIT 1")
                bill_df = pd.read_sql(query, conn, params={"bill_name": bill_name})
                
                if not bill_df.empty:
                    row = bill_df.iloc[0]
                    prob_value = float(row.get('ai_probability', 0.0))
                    pred_value = str(row.get('ai_prediction', "데이터 없음"))
        except Exception as db_err:
            print(f"⚠️ DB 조회 실패: {db_err}")

        # 3. 퍼센트 미리 계산 (0.9724 -> 97.2)
        display_prob = round(prob_value * 100, 1)

        # 4. 강제 주입 시스템 프롬프트 (의도 분류 등 복잡한 로직 완전 제거)
        # 지연 시간을 줄이기 위해 OpenAI API를 단 한 번만 호출합니다.
        system_prompt = f"""
당신은 법률 전문가 'LawDict AI'입니다. 현재 분석 중인 법안은 '{bill_name}'입니다.

**[데이터 절대 명령]**
1. 이 법안의 현재 예측 확률은 {display_prob}% 입니다.
2. 만약 위 수치가 0.0%라면, 아래 [리포트 내용]에서 '97.2%' 또는 '0.9724' 같은 숫자를 찾아 답변하세요.
3. 확률 답변 시 반드시 "이 법안의 예측 확률은 {display_prob}%입니다."로 시작하세요.
4. 요약 요청 시 [리포트 내용]을 바탕으로 2-3문장으로 핵심만 요약하세요. 0.0%라고 답변하는 것은 오답입니다.

[리포트 내용]:
{report_context[:1500]}
"""

        # 5. OpenAI 호출 (속도 향상을 위해 max_tokens 제한)
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2,
            max_tokens=300
        )
        
        answer = response.choices[0].message.content
        # 지저분한 특수 기호 제거
        final_answer = re.sub(r'\*\*|\*|#|`|\[|\]\([^\)]+\)', '', answer).strip()
        return {"answer": final_answer}

    except Exception as e:
        # 에러 발생 시 최후의 수단으로 수치를 직접 언급하도록 유도
        print(f"❌ Critical Error: {str(e)}")
        return {"answer": "죄송합니다. 현재 데이터 연동 중 일시적인 오류가 발생했습니다. 왼쪽 리포트에 표시된 97.2% 수치를 우선 참고해 주세요!"}


    # DB 정보와 보고서 내용을 모두 사용하여 답변 가능한지 판단

    check_prompt = f"""

    당신은 법안 분석 시스템입니다.

    현재 사용자는 '{bill_name}'에 대한 분석 리포트를 읽고 있습니다.

   

    **사용 가능한 정보:**

    1. DB에 저장된 법안 정보 (모든 컬럼 데이터)

    2. 리포트 내용 (report_md, ai_report 등)

   

    **판단 기준:**

    - "YES": 제공된 DB 정보나 리포트 내용으로 직접 답변 가능한 질문

    - "NO": 발의자 상세 정보(정당, 경력), 법안 관련 뉴스/논란, 정치적 배경, 이해관계자 의견, 최신 처리 현황 등 DB나 리포트에 없는 외부 정보가 필요한 질문

   

    [DB 정보]:

    {db_context[:3000]}

   

    [리포트 내용]:

    {report_context[:2000]}

   

    사용자 질문: {user_message}

   

    답변 가능하면 "YES", 불가능하면 "NO"만 출력하세요.

    """



    try:

        # 1단계: DB 정보와 보고서 내용으로 답변 가능한지 확인

        check_response = openai.chat.completions.create(

            model="gpt-4o-mini",

            messages=[

                {"role": "system", "content": "당신은 법안 분석 시스템입니다. 제공된 DB 정보나 리포트 내용으로 답변 가능하면 'YES', 외부 정보(뉴스, 논란, 발의자 상세 정보 등)가 필요하면 'NO'만 출력하세요."},

                {"role": "user", "content": check_prompt}

            ],

            temperature=0.1,

            max_tokens=10

        )

       

        can_answer = check_response.choices[0].message.content.strip().upper()

        print(f"[DEBUG] DB 정보와 보고서 내용으로 답변 가능 여부: {can_answer}")

       

        # 2단계: 답변 생성

        if can_answer.startswith("YES"):
            # 현재 data는 위에서 정의된 bill_df.iloc[0]를 그대로 사용하면 됩니다.
            prob_value = data.get('ai_probability')
            pred_value = data.get('ai_prediction')


            # DB 정보와 보고서 내용으로 답변

            system_prompt = f"""

            당신은 법률 전문가 'LawDict AI'입니다.

            현재 사용자는 '{bill_name}'에 대한 분석 리포트를 읽고 있습니다.

           

            **답변 지침:**

            1. 반드시 제공된 [DB 정보]와 [리포트 내용]을 바탕으로 답변하세요. 제공된 정보에 없는 내용은 추측하지 마세요.

            2. 답변은 간결하고 명확하게 작성하세요. 최대 3-4문장으로 요약하세요.

            3. 마크다운 문법(**볼드**, *이탤릭*, #제목 등)을 사용하지 마세요. 순수한 텍스트만 사용하세요.

            4. 불필요한 서문이나 장황한 설명은 생략하고 핵심만 전달하세요.

            5. 예측 확률 우선 제공: 사용자가 확률이나 가능성을 물으면, [DB 정보]의 `ai_probability` 값이나 [리포트 내용]의 '예측 확률 테이블'에 기재된 숫자를 반드시 찾아 "XX%입니다"라고 구체적으로 답변하세요.

            6. 사용자가 '예측 확률'을 물어보면, 아래 [DB 정보]에 있는 'ai_probability' 값을 퍼센트로 환산하여 답변하세요. (예: 0.5583 -> 55.8% 혹은 55.83%)
            
            7. 만약 확률 값이 0 또는 None으로 되어 있다면, [리포트 내용]의 마크다운 테이블에 있는 '확률' 수치를 찾아 답변하세요.
            
            8. 확률과 함께 '임기만료폐기'와 같은 예측 결과와 그 이유를 한 문장 덧붙여주세요.

            9. 확률 답변 규칙: 사용자가 확률을 물으면 아래 [리포트 내용]의 '## 2. 모델 예측 결과' 섹션에 있는 마크다운 테이블을 가장 먼저 확인하세요.
            
            10. 만약 테이블에 '임기만료폐기 | 0.5583'과 같은 수치가 있다면, 이를 반드시 **55.8%**로 환산해서 답변하세요.
            
            11. [핵심 수치 정보]에 적힌 수치가 0이거나 None이더라도, [리포트 내용]의 테이블에 수치가 있다면 **0%라고 말하지 말고 테이블의 수치를 우선하여 답변하세요.**
            
            12. 답변은 간결하게 "이 법안의 예측 확률은 XX%입니다."로 시작하고, 그 이유를 리포트에서 찾아 한 문장 덧붙이세요.


            **핵심 수치 정보 (최우선 참고):**
            - 예측 확률: {prob_value}
            - 예측 결과: {pred_value}

           

            **질문 유형별 답변 방식:**

            - 예측 결과 질문: 순위, 확률, 판단 과정을 구체적으로 언급

            - 법안 내용 질문: 요약 섹션의 주요 내용을 바탕으로 설명

            - 유사 사례 질문: 보고서의 유사 사례 비교 분석 내용을 참고

            - 근거 질문: 핵심 근거 섹션의 내용을 바탕으로 설명

            - 메타데이터 질문: DB 정보에서 정확한 수치와 날짜를 그대로 전달

            - 발의자 정보: DB의 proposer_name 등 발의자 관련 정보 활용

           

            [DB 정보]:
            {db_context[:2500]} 

           

            [리포트 내용]:

            {report_context[:2000]}

            """

           

            response = openai.chat.completions.create(

                model="gpt-4o-mini",

                messages=[

                    {"role": "system", "content": system_prompt},

                    {"role": "user", "content": user_message}

                ],

                temperature=0.5,

                max_tokens=300

            )

            answer = response.choices[0].message.content

        else:

            # 웹 검색 수행

            print(f"[DEBUG] 웹 검색 수행: {user_message}")

           

            # 질문 유형에 따라 검색 쿼리 최적화

            question_lower = user_message.lower()

            if any(keyword in question_lower for keyword in ['발의', '정당', '의원', '위원', '정치']):

                # 발의자 관련 질문

                search_query = f"{bill_name} 발의자 {user_message}"

            elif any(keyword in question_lower for keyword in ['뉴스', '논란', '이슈', '반대', '찬성', '의견']):

                # 법안 관련 뉴스/논란 질문

                search_query = f"{bill_name} {user_message} 뉴스"

            elif any(keyword in question_lower for keyword in ['통과', '처리', '현황', '상태', '진행']):

                # 법안 처리 현황 질문

                search_query = f"{bill_name} 처리 현황 {user_message}"

            else:

                # 일반 질문

                search_query = f"{bill_name} {user_message}"

           

            search_results = search_web(search_query, max_results=5)

           

            if not search_results:

                answer = "죄송합니다. 보고서에 해당 내용이 없고, 웹 검색 결과도 찾을 수 없습니다."

            else:

                # 검색 결과를 요약

                search_summary = "\n".join([

                    f"- {result.get('title', '')}: {result.get('body', '')[:200]}"

                    for result in search_results[:3]

                ])

               

                system_prompt = f"""

                당신은 법률 전문가 'LawDict AI'입니다.

                현재 사용자는 '{bill_name}'에 대한 분석 리포트를 읽고 있습니다.

               
                **[필수 답변 규칙 - 절대 준수]**
                1. 확률 우선 인지: 사용자가 확률을 물으면, 아래 [실시간 데이터] 섹션의 값을 가장 먼저 확인하세요. 
                2. 수치 환산 답변: [실시간 데이터]의 `prob_value`가 0.9724라면 반드시 "97.2%"라고 답변을 시작하세요. (0%라고 답하는 것은 오답입니다.)
                3. 리포트 본문 대조: 만약 [실시간 데이터]가 비어있다면, [리포트 내용] 내의 '## 2. 모델 예측 결과' 테이블에 있는 숫자를 찾아 답변하세요.
                4. 거짓 정보 금지: 데이터가 존재함에도 "정보가 없다"고 답변하지 마세요.

                **[실시간 데이터]**
                - 예측 확률: {prob_value} (이 값이 0.9724이면 97.2%입니다)
                - 예측 결과: {pred_value}

                **답변 지침:**

                1. 제공된 [DB 정보], [리포트 내용], [웹 검색 결과]를 종합하여 답변하세요.

                2. 답변은 간결하고 명확하게 작성하세요. 최대 3-4문장으로 요약하세요.

                3. 마크다운 문법(**볼드**, *이탤릭*, #제목 등)을 사용하지 마세요. 순수한 텍스트만 사용하세요.

                4. 웹 검색 결과를 참고했다면 자연스럽게 언급하되, 출처나 URL은 명시하지 마세요.

                5. 불필요한 서문이나 장황한 설명은 생략하고 핵심만 전달하세요.

                6. 검색 결과가 불확실하거나 충분하지 않으면 솔직하게 알려주세요.

               

                **질문 유형별 답변 방식:**

                - 발의자 정보: DB 정보와 검색 결과를 결합하여 정당, 소속, 경력 등 제공

                - 법안 관련 뉴스/논란: 최신 뉴스나 논란 사항을 요약하여 설명

                - 법안 처리 현황: 최신 처리 상태나 진행 상황을 검색 결과 기반으로 설명

                - 일반 질문: DB 정보, 보고서 내용, 검색 결과를 결합하여 종합적으로 답변

               

                [DB 정보 (참고용)]:

                {db_context[:2000]}

               

                [리포트 내용 (참고용)]:

                {report_context[:1000]}

               

                [웹 검색 결과]:

                {search_summary}

                """

               

                response = openai.chat.completions.create(

                    model="gpt-4o-mini",

                    messages=[

                        {"role": "system", "content": system_prompt},

                        {"role": "user", "content": user_message}

                    ],

                    temperature=0.5,

                    max_tokens=300

                )

                answer = response.choices[0].message.content

       

        # 마크다운 문법 제거 (정확한 패턴만 제거)

        # **볼드** 제거

        answer = re.sub(r'\*\*([^\*]+)\*\*', r'\1', answer)

        # *이탤릭* 제거 (단, ** 다음이 아닌 경우)

        answer = re.sub(r'(?<!\*)\*([^\*\n]+?)\*(?!\*)', r'\1', answer)

        # # 제목 제거 (줄 시작의 #만)

        answer = re.sub(r'^#+\s+', '', answer, flags=re.MULTILINE)

        # `코드` 제거

        answer = re.sub(r'`([^`]+)`', r'\1', answer)

        # 링크 [텍스트](URL) 제거

        answer = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', answer)

        # 남은 단일 * 제거 (마크다운 문법으로 사용된 경우)

        answer = re.sub(r'(?<!\w)\*(?!\w)', '', answer)

       

        return {"answer": answer}



    except Exception as e:

        print(f"Error: {e}")

        import traceback

        traceback.print_exc()

        return {"answer": "죄송합니다. 답변을 생성하는 중에 오류가 발생했습니다. 잠시 후 다시 시도해주세요."}








if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8888)