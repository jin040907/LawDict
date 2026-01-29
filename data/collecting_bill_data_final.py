# -*- coding: utf-8 -*-
"""
[ALLBILL 단건조회(BILL_NO 필수) 기반] 의안 수집기 (초기 시드 + 이후 최근 5일 재수집)

요구사항 반영
- 요청 인자: BILL_NO만 사용 (ERROR-300 대응)
- 최초 실행(누적 테이블 비어있음): seed_bill_no=2215960 부터 시작
- 이후 실행: (오늘-5일) 범위의 데이터 중 MIN(bill_no)부터 시작해서 최신까지 스캔
- 스캔 종료: 연속 miss(응답없음/row=[]) MAX_CONSECUTIVE_MISS 번이면 최신 도달로 보고 종료
- 크롤링: Requests → 실패/요약짧음 → Selenium 폴백
- 임베딩: bge-m3 (pgvector vector(1024) 가정, 불일치 시 에러)
- 적재:
  1) staging(all_bill_raw_storage): TRUNCATE 후 이번 실행분만 적재
  2) integrated(all_bill_integrated): bill_no 기준 UPSERT(누적 유지)  ← (권장)
     ※ integrated가 없거나 컬럼 구조가 다르면 upsert 쿼리만 조정하면 됩니다.

필수 준비
- /home/cginside19/intern/model/.env 에 ASSEMBLY_API_KEY=... 형태로 저장
- DB에 pgvector extension 설치되어 있어야 함(이미 vector 쓰는 상태라면 OK)
"""

import os
import re
import time
import datetime as dt
from typing import Tuple, Optional, Dict, Any, List

import requests
import pandas as pd
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, text
from psycopg2.extras import execute_values

import torch
from sentence_transformers import SentenceTransformer

# Selenium 폴백용
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

from dotenv import load_dotenv
from pathlib import Path

import datetime as dt

# =========================
# 0) 설정
# =========================
# (경고 대응) PYTORCH_CUDA_ALLOC_CONF → PYTORCH_ALLOC_CONF
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

DB_URL_BILL = "postgresql://cginside19:1234@localhost:5432/bill_db"

DOTENV_PATH = Path("/home/cginside19/intern/model/.env")
load_dotenv(dotenv_path=DOTENV_PATH, override=True)

API_KEY = os.getenv("OPEN_ASSEMBLY_API_KEY")
if not API_KEY:
    raise RuntimeError("OPEN_ASSEMBLY_API_KEY가 .env에 없거나 로드되지 않았습니다.")

STAGING_TABLE = "all_bill_raw_storage"
INTEGRATED_TABLE = "all_bill_integrated"

# 최초 부트스트랩 시작점
SEED_BILL_NO = 2215960

# 이후 매 실행 때 '최근 며칠' 구간에서 MIN(bill_no)부터 재수집
RECENT_DAYS_RESCAN = 5

# 최신 도달 판단: 연속 miss가 이 횟수 넘으면 중단
MAX_CONSECUTIVE_MISS = 10

# API 과호출 방지
SLEEP_SEC = 0.15

# pgvector 차원 (bge-m3 기본 1024 가정)
EXPECTED_EMB_DIM = 1024

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# 모델 캐시
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}


def get_model(device: str) -> SentenceTransformer:
    if device not in _MODEL_CACHE:
        print(f"// [Model] Loading BAAI/bge-m3 on {device} ...")
        _MODEL_CACHE[device] = SentenceTransformer("BAAI/bge-m3", device=device)
    return _MODEL_CACHE[device]


# =========================
# 1) 가공 유틸
# =========================
def parse_amendment_type(bill_nm: str) -> str:
    for t in ["전부개정", "일부개정", "헌법개정", "폐지"]:
        if t in (bill_nm or ""):
            return t
    return "제정"


from urllib.parse import urljoin

LIKMS_BASE = "https://likms.assembly.go.kr"
LIKMS_BILL_BASE = "https://likms.assembly.go.kr/bill"  # 미리보기 data-url이 /bi/... 형태일 때 필요

def extract_summary_budget_and_pdf_links(html: str, page_url: Optional[str] = None):
    soup = BeautifulSoup(html, "html.parser")

    # 1) summary
    summary = ""
    pre = soup.select_one("pre#prntSummary")
    if pre:
        summary = pre.get_text("\n", strip=True)

    # 2) budget(비용추계서 존재 여부)
    has_budget = 0
    if re.search(r"(?<!미첨부\s)비용추계서", html) and ("미첨부 사유서" not in html):
        has_budget = 1
    if soup.find("a", title=re.compile(r"^비용추계서")):
        has_budget = 1

    # 3) PDF 다운로드 링크 (a.icon_pdf href="https://likms.assembly.go.kr/filegate/servlet/FileGate?....")
    pdf_download_url = None
    a_pdf = soup.select_one("a.icon_pdf[href]")
    if a_pdf:
        href = a_pdf.get("href", "").strip()
        if href:
            # 절대/상대 모두 대응
            if page_url:
                pdf_download_url = urljoin(page_url, href)
            else:
                pdf_download_url = urljoin(LIKMS_BASE, href)

    # 4) PDF 미리보기 링크
    # 캡처처럼 a.icon_preview data-url="/bi/common/preview/pdfPreview.do?bookId=...&section=bill&filetype=p"
    pdf_preview_url = None
    a_prev = soup.select_one("a.icon_preview[data-url]")
    if a_prev:
        data_url = (a_prev.get("data-url") or "").strip()
        if data_url:
            # data-url은 /bi/... 이고, 실제 동작 URL은 /bill/bi/... 로 붙는 케이스가 많음 (질문에 준 예시도 /bill/bi/..)
            # -> https://likms.assembly.go.kr/bill + data_url 로 구성
            pdf_preview_url = urljoin(LIKMS_BILL_BASE + "/", data_url.lstrip("/"))

    return summary, has_budget, pdf_download_url, pdf_preview_url



def to_pgvector_str(vec: List[float]) -> str:
    # 공백 최소화 + 고정 소수점(파싱 안정)
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def get_committee_code_map(engine) -> Dict[str, str]:
    """
    all_bill_integrated에서 curr_committee_name→curr_committee_code 매핑 재사용
    (없어도 동작: None으로 들어감)
    """
    try:
        with engine.connect() as conn:
            df = pd.read_sql(
                f"""
                SELECT DISTINCT curr_committee_name, curr_committee_code
                FROM {INTEGRATED_TABLE}
                WHERE curr_committee_code IS NOT NULL
                """,
                conn,
            )
        return dict(zip(df["curr_committee_name"], df["curr_committee_code"]))
    except Exception as e:
        print(f"// [WARN] committee_code_map load failed: {e}")
        return {}


# =========================
# 2) 크롤링 (Requests → Selenium 폴백)
# =========================
def _get_driver() -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=opts)


def crawl_bill_detail_hybrid(url: Optional[str]):
    if not url:
        return "", 0, None, None

    # 1) requests
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            s, b, pdf_d, pdf_p = extract_summary_budget_and_pdf_links(resp.text, page_url=url)
            if s and len(s) > 20:
                return s, b, pdf_d, pdf_p
            # 요약이 짧아도 pdf 링크는 이미 뽑혔을 수 있으니, selenium 넘어가기 전에 일단 반환할지 선택 가능
            # 여기선 기존 정책대로 summary 부족하면 selenium fallback
    except Exception:
        pass

    # 2) selenium fallback
    driver = None
    try:
        driver = _get_driver()
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "prntSummary"))
        )
        s, b, pdf_d, pdf_p = extract_summary_budget_and_pdf_links(driver.page_source, page_url=url)
        return s, b, pdf_d, pdf_p
    except Exception:
        return "", 0, None, None
    finally:
        if driver:
            driver.quit()



# =========================
# 3) ALLBILL 단건 조회 (BILL_NO만)
# =========================
def fetch_one_bill_allbill(bill_no: int) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    returns: (rows, result_obj)
    - rows: 성공 시 ALLBILL[1].row 리스트
    - result_obj: RESULT가 있는 경우(에러/안내) 그 객체
    """
    params = {
        "KEY": API_KEY,
        "Type": "json",
        "pIndex": 1,
        "pSize": 100,
        "BILL_NO": str(bill_no),
    }
    resp = requests.get(
        "https://open.assembly.go.kr/portal/openapi/ALLBILL",
        params=params,
        headers=HEADERS,
        timeout=15,
    )
    data = resp.json()

    if "RESULT" in data:
        return [], data["RESULT"]

    rows = data.get("ALLBILL", [{}, {"row": []}])[1].get("row", [])
    return rows, None


# =========================
# 4) 시작 bill_no 결정 로직
# =========================
def get_start_bill_no(engine, seed: int, recent_days: int) -> int:
    """
    - integrated에 데이터가 있으면: 최근 recent_days 구간의 MIN(bill_no)
    - 없으면: seed
    """
    try:
        with engine.connect() as conn:
            # integrated가 비었거나 propose_dt가 없으면 None
            v_any = conn.execute(text(f"SELECT MAX(bill_no) FROM {INTEGRATED_TABLE}")).scalar()
            if v_any is None:
                print(f"// [Start] integrated empty → seed={seed}")
                return seed

            # 최근 N일 구간의 최소 bill_no로 되감기(재수집)
            q = text(
                f"""
                SELECT MIN(bill_no)
                FROM {INTEGRATED_TABLE}
                WHERE propose_dt >= (CURRENT_DATE - (:days || ' days')::interval)
                """
            )
            v_min = conn.execute(q, {"days": recent_days}).scalar()
            if v_min is None:
                # 최근 구간이 비면 그냥 최신+1부터 하든지, seed부터 하든지 선택인데
                # 여기서는 "최신+1"보다 "seed"가 너무 옛날이라 비효율이라, 최신+1 추천
                start = int(v_any) + 1
                print(f"// [Start] no rows in recent {recent_days}d → start=max+1={start}")
                return start

            start = int(v_min)
            print(f"// [Start] recent {recent_days}d MIN(bill_no)={start}")
            return start
    except Exception as e:
        print(f"// [WARN] get_start_bill_no failed: {e} → seed={seed}")
        return seed


# =========================
# 5) Staging 테이블 보장
# =========================
def ensure_staging_table(engine) -> None:
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {STAGING_TABLE} (
        bill_no BIGINT PRIMARY KEY,
        bill_id TEXT,
        bill_name TEXT,
        propose_dt DATE,
        elapsed_days INT,
        proposer_count_est INT,
        proposer_kind TEXT,
        budget INT,
        curr_committee_name TEXT,
        curr_committee_code TEXT,
        is_alternative INT,
        amendment_type TEXT,
        summary TEXT,
        proposer_name TEXT,
        ord INT,
        summary_embedding vector({EXPECTED_EMB_DIM}),
        pdf_url TEXT,
        pdf_preview_url TEXT,
        bill_url TEXT
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


# =========================
# 6) 적재 (staging TRUNCATE + integrated UPSERT)
# =========================
def load_to_staging(engine, df: pd.DataFrame, cols: List[str]) -> None:
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {STAGING_TABLE}"))
        raw_conn = conn.connection
        with raw_conn.cursor() as cur:
            insert_sql = f"INSERT INTO {STAGING_TABLE} ({', '.join(cols)}) VALUES %s"
            execute_values(cur, insert_sql, [tuple(x) for x in df[cols].values], page_size=500)


def upsert_to_integrated(engine, df: pd.DataFrame) -> None:
    """
    integrated가 이미 있고 컬럼이 존재한다는 가정.
    컬럼이 더 많아도 상관없고, 아래 명시 컬럼만 업데이트.
    """
    cols = [
        "bill_no",
        "bill_id",
        "bill_name",
        "propose_dt",
        "elapsed_days",
        "proposer_count_est",
        "proposer_kind",
        "budget",
        "curr_committee_name",
        "curr_committee_code",
        "is_alternative",
        "amendment_type",
        "summary",
        "proposer_name",
        "ord",
        "summary_embedding",
        "pdf_url",
        "pdf_preview_url",
        "bill_url"
    ]

    # 임시 테이블로 올린 뒤 SQL로 UPSERT (컬럼 순서/타입 안정)
    tmp = "_tmp_upsert_allbill"

    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {tmp}"))
        conn.execute(text(f"CREATE TEMP TABLE {tmp} (LIKE {STAGING_TABLE} INCLUDING ALL) ON COMMIT DROP"))

        raw_conn = conn.connection
        with raw_conn.cursor() as cur:
            insert_sql = f"INSERT INTO {tmp} ({', '.join(cols)}) VALUES %s"
            execute_values(cur, insert_sql, [tuple(x) for x in df[cols].values], page_size=500)

        # integrated가 bill_no 유니크/PK라는 가정
        upsert_sql = text(
            f"""
            INSERT INTO {INTEGRATED_TABLE} ({', '.join(cols)})
            SELECT {', '.join(cols)} FROM {tmp}
            ON CONFLICT (bill_no)
            DO UPDATE SET
                bill_id = EXCLUDED.bill_id,
                bill_name = EXCLUDED.bill_name,
                propose_dt = EXCLUDED.propose_dt,
                elapsed_days = EXCLUDED.elapsed_days,
                proposer_count_est = EXCLUDED.proposer_count_est,
                proposer_kind = EXCLUDED.proposer_kind,
                budget = EXCLUDED.budget,
                curr_committee_name = EXCLUDED.curr_committee_name,
                curr_committee_code = EXCLUDED.curr_committee_code,
                is_alternative = EXCLUDED.is_alternative,
                amendment_type = EXCLUDED.amendment_type,
                summary = EXCLUDED.summary,
                proposer_name = EXCLUDED.proposer_name,
                ord = EXCLUDED.ord,
                summary_embedding = EXCLUDED.summary_embedding,
                pdf_url = EXCLUDED.pdf_url,
                pdf_preview_url = EXCLUDED.pdf_preview_url,
                bill_url = EXCLUDED.bill_url
            """
        )
        conn.execute(upsert_sql)


# =========================
# 7) 메인 실행
# =========================
def run():
    # 코드의 가장 윗부분 또는 시작 지점
    print(f"// [SYSTEM] Job Started at: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    engine = create_engine(DB_URL_BILL)
    ensure_staging_table(engine)
    comm_map = get_committee_code_map(engine)

    start_bill_no = get_start_bill_no(
        engine=engine,
        seed=SEED_BILL_NO,
        recent_days=RECENT_DAYS_RESCAN,
    )

    print(f"// [RUN] start_bill_no={start_bill_no} (request param: BILL_NO only)")
    miss = 0
    bill_no = start_bill_no

    collected: List[Dict[str, Any]] = []
    last_hit_bill_no: Optional[int] = None

    while miss < MAX_CONSECUTIVE_MISS:
        try:
            rows, result_obj = fetch_one_bill_allbill(bill_no)

            # RESULT가 오는 경우(에러/안내). "해당 데이터 없음"도 RESULT로 올 수 있음.
            if result_obj is not None and not rows:
                miss += 1
                if miss % 10 == 0:
                    code = result_obj.get("CODE")
                    msg = result_obj.get("MESSAGE")
                    print(f"// [MISS] bill_no={bill_no} miss={miss}/{MAX_CONSECUTIVE_MISS} RESULT={code} {msg}")
                bill_no += 1
                time.sleep(SLEEP_SEC)
                continue

            if not rows:
                miss += 1
                if miss % 10 == 0:
                    print(f"// [MISS] bill_no={bill_no} miss={miss}/{MAX_CONSECUTIVE_MISS} (empty rows)")
                bill_no += 1
                time.sleep(SLEEP_SEC)
                continue

            # hit
            miss = 0
            last_hit_bill_no = bill_no

            # ALLBILL row는 보통 1건이지만, 안전하게 첫 row 사용
            r0 = rows[0]

            bill_nm = r0.get("BILL_NM", "") or ""
            bill_id = r0.get("BILL_ID")
            ppsl_dt_str = r0.get("PPSL_DT", "") or ""
            propose_dt = dt.datetime.strptime(ppsl_dt_str, "%Y-%m-%d").date() if ppsl_dt_str else None

            link_url = r0.get("LINK_URL")
            summary, budget, pdf_url, pdf_preview_url = crawl_bill_detail_hybrid(link_url)


            ppsr_nm_raw = r0.get("PPSR_NM", "") or ""
            count_match = re.search(r"등\s*(\d+)\s*인", ppsr_nm_raw)
            proposer_count = int(count_match.group(1)) + 1 if count_match else 1
            proposer_name = re.split(r"의원|등", ppsr_nm_raw)[0].strip()

            jrcmit_nm = r0.get("JRCMIT_NM")
            ord_val = int(re.sub(r"[^0-9]", "", r0.get("ERACO", "0") or "0") or 0)

            collected.append(
                {
                    "bill_no": int(bill_no),
                    "bill_id": bill_id,
                    "bill_name": bill_nm,
                    "propose_dt": propose_dt,
                    "elapsed_days": (dt.date.today() - propose_dt).days if propose_dt else 0,
                    "proposer_count_est": proposer_count,
                    "proposer_kind": r0.get("PPSR_KND"),
                    "budget": int(budget) if budget is not None else 0,
                    "curr_committee_name": jrcmit_nm,
                    "curr_committee_code": comm_map.get(jrcmit_nm),
                    "is_alternative": 1 if "(대안)" in bill_nm else 0,
                    "amendment_type": parse_amendment_type(bill_nm),
                    "summary": summary,
                    "proposer_name": proposer_name,
                    "ord": ord_val,
                    "summary_embedding": None,  # 아래에서 채움
                    "pdf_url": pdf_url,
                    "pdf_preview_url": pdf_preview_url,
                    "bill_url": link_url
                }
            )

            if len(collected) % 10 == 0:
                print(f"// [HIT] bill_no={bill_no} collected={len(collected)}")

            bill_no += 1
            time.sleep(SLEEP_SEC)

        except Exception as e:
            miss += 1
            if miss % 10 == 0:
                print(f"// [WARN] bill_no={bill_no} err={e} miss={miss}/{MAX_CONSECUTIVE_MISS}")
            bill_no += 1
            time.sleep(SLEEP_SEC)

    if not collected:
        print("// [INFO] 수집된 데이터가 없습니다.")
        return

    df = pd.DataFrame(collected)

    # 임베딩 컨텍스트(검색 품질 개선)
    contexts = [
        f"{r['bill_name']}\n소관위: {r['curr_committee_name'] or '미정'}\n요약: {r['summary'] or '요약 정보 없음'}"
        for _, r in df.iterrows()
    ]

    print(f"// [Embedding] n={len(contexts)}")
    try:
        emb = get_model("cuda").encode(contexts, batch_size=16, normalize_embeddings=True)
    except Exception as e:
        print(f"// [WARN] GPU encode failed: {e} → CPU fallback")
        emb = get_model("cpu").encode(contexts, batch_size=8, normalize_embeddings=True)

    dim = len(emb[0])
    if dim != EXPECTED_EMB_DIM:
        raise ValueError(f"Embedding dim mismatch: expected={EXPECTED_EMB_DIM}, got={dim}")

    df["summary_embedding"] = [to_pgvector_str(v.tolist()) for v in emb]

    cols = [
        "bill_no",
        "bill_id",
        "bill_name",
        "propose_dt",
        "elapsed_days",
        "proposer_count_est",
        "proposer_kind",
        "budget",
        "curr_committee_name",
        "curr_committee_code",
        "is_alternative",
        "amendment_type",
        "summary",
        "proposer_name",
        "ord",
        "summary_embedding",
        "pdf_url",
        "pdf_preview_url",
        "bill_url"
    ]

    # 1) staging 적재(이번 실행분만)
    print(f"// [DB] staging TRUNCATE+INSERT rows={len(df)}")
    load_to_staging(engine, df, cols)

    # 2) integrated upsert(누적)
    # integrated 테이블이 없다면 여기서 에러가 날 수 있습니다.
    print(f"// [DB] integrated UPSERT rows={len(df)}")
    upsert_to_integrated(engine, df)

    print(f"// ✅ [DONE] collected={len(df)} last_hit_bill_no={last_hit_bill_no} miss_end={miss}/{MAX_CONSECUTIVE_MISS}")
    print(f"// [SYSTEM] Job Finished at: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    run()
    