# backend/similar_cases_pgvector.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from sqlalchemy import text


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _cosine_distance_to_similarity(d: Optional[float]) -> Optional[float]:
    """
    pgvector <=> 는 cosine distance.
    보통 similarity = 1 - distance 로 사용(0~1 근방).
    안전하게 0~1로 clamp.
    """
    if d is None:
        return None
    sim = 1.0 - d
    if sim < 0.0:
        sim = 0.0
    if sim > 1.0:
        sim = 1.0
    return sim


def _l2_distance_to_similarity(d: Optional[float]) -> Optional[float]:
    """
    L2는 스케일이 제각각이라 0~1로 매핑용 휴리스틱.
    """
    if d is None:
        return None
    return 1.0 / (1.0 + d)


def fetch_similar_cases(
    *,
    engine,
    table_full: str,
    bill_id: str,
    top_k: int = 10,
    same_committee: bool = False,
    embedding_col: str = "summary_embedding",
    committee_col: str = "curr_committee_name",
    include_raw: bool = True,
) -> Dict[str, Any]:
    """
    pgvector 기반 유사사례 조회.
    반환 포맷은 report_builder_with_llm._render_similar_cases 가 기대하는 형태에 맞춤.

    rows 각 원소는 최소:
      - bill_id, bill_name
      - y_raw (실제결과)
      - similarity (0~1 근방)
    을 포함하도록 구성.
    """
    filters: List[str] = []
    if same_committee:
        filters = ["same_committee", committee_col]

    # cosine 먼저 시도
    sql_cos = text(f"""
        WITH q AS (
            SELECT bill_id, {embedding_col} AS emb, {committee_col} AS q_committee
            FROM {table_full}
            WHERE bill_id::text = :bill_id
              AND {embedding_col} IS NOT NULL
            LIMIT 1
        )
        SELECT
            t.*,
            (t.{embedding_col} <=> q.emb) AS distance
        FROM {table_full} t
        JOIN q ON TRUE
        WHERE t.{embedding_col} IS NOT NULL
          AND t.bill_id <> q.bill_id
          AND (:same_committee = FALSE OR t.{committee_col} = q.q_committee)
        ORDER BY t.{embedding_col} <=> q.emb
        LIMIT :k
    """)

    # L2 폴백
    sql_l2 = text(f"""
        WITH q AS (
            SELECT bill_id, {embedding_col} AS emb, {committee_col} AS q_committee
            FROM {table_full}
            WHERE bill_id::text = :bill_id
              AND {embedding_col} IS NOT NULL
            LIMIT 1
        )
        SELECT
            t.*,
            (t.{embedding_col} <-> q.emb) AS distance
        FROM {table_full} t
        JOIN q ON TRUE
        WHERE t.{embedding_col} IS NOT NULL
          AND t.bill_id <> q.bill_id
          AND (:same_committee = FALSE OR t.{committee_col} = q.q_committee)
        ORDER BY t.{embedding_col} <-> q.emb
        LIMIT :k
    """)

    params = {"bill_id": str(bill_id), "k": int(top_k), "same_committee": bool(same_committee)}

    metric = "cosine(<=>)"
    rows: List[Dict[str, Any]] = []

    with engine.begin() as conn:
        # 기준 bill 존재/임베딩 존재 체크
        chk = conn.execute(
            text(f"SELECT {embedding_col} IS NOT NULL FROM {table_full} WHERE bill_id::text = :bill_id"),
            {"bill_id": str(bill_id)},
        ).scalar()

        if chk is None:
            raise ValueError(f"bill_id={bill_id} 가 테이블에 없습니다.")
        if chk is False:
            raise ValueError(f"bill_id={bill_id} 는 {embedding_col} 이 NULL 입니다. (백필 누락)")

        try:
            result = conn.execute(sql_cos, params).mappings().all()
        except Exception as e:
            metric = "L2(<->)"
            result = conn.execute(sql_l2, params).mappings().all()

    for r in result:
        raw = dict(r)

        dist = _to_float(raw.get("distance"))
        if metric.startswith("cosine"):
            sim = _cosine_distance_to_similarity(dist)
        else:
            sim = _l2_distance_to_similarity(dist)

        # 보고서 렌더러가 기대하는 키로 맞춤
        item: Dict[str, Any] = {
            "bill_id": raw.get("bill_id"),
            "bill_name": raw.get("bill_name"),
            "y_raw": raw.get("general_result"),   # 실제 결과
            "similarity": sim,                    # 0~1 근방
            "score": sim,                         # 호환용
            "metric": metric,
        }

        if include_raw:
            # 예측/차이계산에 쓰라고 raw를 숨겨서 넘김(나중에 make_sample_input에서 제거)
            # embedding은 너무 크니 제거
            raw.pop(embedding_col, None)
            item["_raw"] = raw

        rows.append(item)

    return {
        "method": f"pgvector {metric}",
        "filters": filters,
        "top_k": int(top_k),
        "rows": rows,
    }
