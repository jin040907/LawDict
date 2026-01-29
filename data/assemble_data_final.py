import pandas as pd
import datetime as dt
from sqlalchemy import create_engine, text

# =========================
# 0) 설정
# =========================
BILL_DB_URL = "postgresql://cginside19:1234@localhost:5432/bill_db"
MEMBER_DB_URL = "postgresql://cginside19:1234@localhost:5432/member"

# 목적지 테이블을 직접 수정했습니다.
TARGET_TABLE = "final_training_data_copy"

# 최종 결과 컬럼 규격 (term 제외, 총 30개 컬럼)
FINAL_COLS = [
    "bill_id", "bill_name", "propose_dt", "elapsed_days", "proposer_count_est",
    "proposer_kind", "budget", "curr_committee_code", "curr_committee_name",
    "is_alternative", "amendment_type", "bill_no", "summary", "proposer_name",
    "ord", "summary_embedding", "naas_cd", "naas_nm",
    "reelection_count", "ntr_div", "election_type", "district_at_term",
    "party_at_term", "party_seats", "is_negotiation_group_party",
    "has_committee", "committee_at_term", "n_negotiation_groups",
    "social_issue_yn", "social_issue_score"
]

def run_integrator():
    # 시작 시간 기록
    print(f"// [SYSTEM] Job Started at: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    bill_engine = create_engine(BILL_DB_URL)
    member_engine = create_engine(MEMBER_DB_URL)

    print("// [Step 1] DB 마스터 정보 로드 및 전처리...")
    raw_bills = pd.read_sql("SELECT * FROM all_bill_raw_storage", bill_engine)
    raw_bills['propose_dt'] = pd.to_datetime(raw_bills['propose_dt']).dt.date
    
    mp_master = pd.read_sql("""
        SELECT m.mp_id as naas_cd, m.mp_name, m.gender as ntr_div, 
               mc.reelection_count, mc.election_div as election_type, 
               p.party_name as party_at_term, mc.party_id
        FROM mp_22 m
        LEFT JOIN mp_22_career mc ON m.mp_id = mc.mp_id
        LEFT JOIN party p ON mc.party_id = p.party_id
    """, member_engine)
    
    pres_master = pd.read_sql("""
        SELECT p.start_date, p.finish_date, pt.party_name as ruling_party
        FROM president p
        JOIN party pt ON p.party_id = pt.party_id
    """, member_engine)
    pres_master['start_date'] = pd.to_datetime(pres_master['start_date']).dt.date
    pres_master['finish_date'] = pd.to_datetime(pres_master['finish_date']).dt.date

    comm_data = pd.read_sql("SELECT mp_id, committee_name FROM mp_22_committee", member_engine)
    mp_comm_map = comm_data.groupby('mp_id')['committee_name'].apply(list).to_dict()

    print("// [Step 2] 필드 가공 및 로직 계산 중...")
    df = pd.merge(raw_bills, mp_master, left_on='proposer_name', right_on='mp_name', how='left')

    def get_party_status(row):
        bill_dt = row['propose_dt']
        if pd.isna(bill_dt) or pd.isna(row['party_at_term']): return None
        match = pres_master[(pres_master['start_date'] <= bill_dt) & (pres_master['finish_date'] >= bill_dt)]
        if not match.empty:
            return '여당' if row['party_at_term'] == match.iloc[0]['ruling_party'] else '야당'
        return '야당'

    def check_committee_match(row):
        target_comm = row['curr_committee_name']
        mp_id = row['naas_cd']
        if pd.isna(target_comm) or pd.isna(mp_id): return 0
        my_committees = mp_comm_map.get(mp_id, [])
        return 1 if target_comm in my_committees else 0

    df['district_at_term'] = df.apply(get_party_status, axis=1)
    df['committee_at_term'] = df.apply(check_committee_match, axis=1)
    df['has_committee'] = df['naas_cd'].apply(lambda x: 1 if x in mp_comm_map else 0)
    df['naas_nm'] = df['mp_name']
    df['party_seats'] = df['party_at_term'].map(mp_master.groupby('party_at_term').size().to_dict())
    
    nego = pd.read_sql("SELECT terms, party_id FROM negotiation_body", member_engine)
    nego_set = set(zip(nego['terms'], nego['party_id']))
    df['is_negotiation_group_party'] = df.apply(lambda x: 1 if (x['ord'], x['party_id']) in nego_set else 0, axis=1)
    df['n_negotiation_groups'] = df['ord'].map(nego.groupby('terms').size().to_dict())
    
    df['social_issue_yn'] = None
    df['social_issue_score'] = None

    df_final = df[FINAL_COLS].where(pd.notnull(df[FINAL_COLS]), None)

    print(f"// [Step 3] {TARGET_TABLE} UPSERT 실행 중...")
    df_final.to_sql('tmp_integrator', bill_engine, if_exists='replace', index=False)
    
    # 타입 캐스팅 및 쿼리 빌드 (목적지 테이블 구조 반영)
    select_parts = []
    for c in FINAL_COLS:
        if c == "summary_embedding":
            select_parts.append("summary_embedding::vector")
        elif c == "budget":
            # bigint -> integer -> boolean 2단계 캐스팅
            select_parts.append("budget::integer::boolean")
        elif c == "social_issue_yn":
            select_parts.append("social_issue_yn::integer")
        elif c == "social_issue_score":
            select_parts.append("social_issue_score::numeric")
        elif c in ["district_at_term", "party_at_term", "bill_id", "proposer_name", "naas_cd", "naas_nm", "bill_name", "summary"]:
            select_parts.append(f"{c}::text")
        else:
            select_parts.append(c)

    # 업데이트 대상 컬럼 (기존 데이터 보존을 위해 수집 정보 위주로만 지정)
    update_cols = [
        "bill_id", "bill_name", "propose_dt", "elapsed_days", "proposer_count_est",
        "proposer_kind", "budget", "curr_committee_code", "curr_committee_name",
        "is_alternative", "amendment_type", "summary", "proposer_name",
        "ord", "summary_embedding", "naas_cd", "naas_nm", "reelection_count",
        "ntr_div", "election_type", "district_at_term", "party_at_term"
    ]
    update_parts = [f"{col} = EXCLUDED.{col}" for col in update_cols]

    upsert_sql = text(f"""
        INSERT INTO {TARGET_TABLE} ({", ".join(FINAL_COLS)})
        SELECT {", ".join(select_parts)} FROM tmp_integrator
        ON CONFLICT (bill_no) 
        DO UPDATE SET {", ".join(update_parts)};
    """)

    with bill_engine.begin() as conn:
        conn.execute(upsert_sql)
        conn.execute(text("DROP TABLE tmp_integrator;"))

    print(f"// ✅ [DONE] 모든 데이터 통합 및 {TARGET_TABLE} 업데이트 완료.")
    print(f"// [SYSTEM] Job Finished at: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    run_integrator()