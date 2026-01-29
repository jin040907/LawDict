# scripts/make_sample_input.py
from __future__ import annotations

import os
import json
import argparse
from types import SimpleNamespace
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional, List

import pandas as pd
from sqlalchemy import create_engine, text

from backend.model_runtime import ModelRuntime
from backend.similar_cases_pgvector import fetch_similar_cases
from backend.report_insights import build_report_insights
from scripts.parse_training_log import parse_training_log


# ✅ 샘플 입력(json/payload)에서 제외할 컬럼들(컬럼명)
# - summary_embedding: DB에서 vector/list 등으로 들어오는 경우가 많아 샘플 생성/가공 시 문제 유발 가능
# - proc_stage_cd/proc_dt/term: 모델명(noProc_noTerm) 기준으로도 보통 불필요(원하면 여기서만 빼면 됨)
# - lwcmt_meeting_count / law_committee_meeting_count: "법사위 회의수" (모델/보고서 모두에서 제거 목적)
# ⚠️ social_issue_yn / social_issue_score 는 여기 넣지 않음 (모델/보고서 근거에 쓰려면 유지)
DROP_COLS = {
    "summary_embedding",
    "proc_stage_cd",
    "proc_dt",
    "term",
    # ✅ 법사위 회의수(모델/보고서 모두에서 제거)
    "lwcmt_meeting_count",
    "law_committee_meeting_count",
}


# ---------- JSON safe ----------
def _jsonable(v):
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, Decimal):
        try:
            return float(v)
        except Exception:
            return str(v)
    return str(v)


def fetch_bill_raw(engine, table_full: str, bill_id: str) -> dict:
    q = text(
        f"""
        SELECT *
        FROM {table_full}
        WHERE bill_id::text = :bill_id
        LIMIT 1
    """
    )
    with engine.begin() as conn:
        row = conn.execute(q, {"bill_id": str(bill_id)}).mappings().fetchone()
        if not row:
            raise RuntimeError(f"bill_id not found in {table_full}: {bill_id}")
        raw = dict(row)

    # ✅ 너무 큰 컬럼/불필요 컬럼 제거(있으면)
    out = {}
    for k, v in raw.items():
        if k in DROP_COLS:
            continue
        out[k] = _jsonable(v)
    return out


def normalize_bill(raw: dict) -> dict:
    def pick(*cands):
        for c in cands:
            if c in raw and raw[c] not in (None, ""):
                return raw[c]
        return None

    bill = {}
    bill["bill_id"] = pick("bill_id")
    bill["bill_no"] = pick("bill_no")
    bill["bill_name"] = pick("bill_name", "bill_nm")
    bill["propose_dt"] = pick("propose_dt", "ppsl_dt")
    bill["committee_name"] = pick("committee_name", "curr_committee_name")
    bill["proposer_name"] = pick("proposer_name", "ppsr_nm", "proposer_nm")
    bill["proposer_count_est"] = pick("proposer_count_est", "proposer_count")
    bill["amendment_type"] = pick("amendment_type")
    # proposer party at the time of the term (used in report section 1)
    bill["party_at_term"] = pick("party_at_term", "party")
    bill["budget_yn"] = pick("budget_yn", "budget")
    bill["elapsed_days"] = pick("elapsed_days")
    bill["committee_meeting_count"] = pick("committee_meeting_count")

    # ✅ 법사위 회의수는 payload/보고서에서 제외
    # bill["law_committee_meeting_count"] = pick("law_committee_meeting_count", "lwcmt_meeting_count")

    bill["summary"] = pick("summary")
    return bill


def _find_model_file(model_dir: Path, prefixes_lower: List[str]) -> Optional[str]:
    """
    prefixes_lower 예:
      ["cbm_step1_", "step1_"] 처럼 여러 prefix를 받음
    """
    files = []
    for p in model_dir.glob("*.cbm"):
        n = p.name.lower()
        if any(n.startswith(pref) for pref in prefixes_lower):
            files.append(p)
    files = sorted(files, key=lambda x: x.name)
    return str(files[-1].resolve()) if files else None


def _ensure_thresholds_json(model_dir: Path, training_log_path: Optional[str]) -> str:
    cand = []
    for p in model_dir.glob("*.json"):
        n = p.name.lower()
        if "threshold" in n or "thr" in n:
            cand.append(p)
    if cand:
        return str(sorted(cand)[-1].resolve())

    out_path = model_dir / "thresholds.json"

    if training_log_path and os.path.exists(training_log_path):
        log_text = Path(training_log_path).read_text(encoding="utf-8", errors="ignore")
        parsed = parse_training_log(log_text)
        saved = parsed.get("saved_thresholds_dict")
        if isinstance(saved, dict) and saved:
            minimal = {
                "thr_fail_step1": saved.get("thr_fail_step1", 0.5),
                "thr_non_alt_step2a": saved.get("thr_non_alt_step2a", 0.5),
                "thr_orig_step2b": saved.get("thr_orig_step2b", 0.5),
                "thr_alt_step2c": saved.get("thr_alt_step2c", 0.5),
            }
            out_path.write_text(json.dumps(minimal, ensure_ascii=False, indent=2), encoding="utf-8")
            return str(out_path.resolve())

    minimal = {
        "thr_fail_step1": 0.5,
        "thr_non_alt_step2a": 0.5,
        "thr_orig_step2b": 0.5,
        "thr_alt_step2c": 0.5,
    }
    out_path.write_text(json.dumps(minimal, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path.resolve())


def build_paths(model_dir: str, training_log: Optional[str]) -> SimpleNamespace:
    md = Path(model_dir).resolve()
    if not md.exists():
        raise RuntimeError(f"--model_dir not found: {md}")

    # ✅ cbm_step* / step* 둘 다 지원
    step1 = _find_model_file(md, ["cbm_step1_", "step1_"])
    step2a = _find_model_file(md, ["cbm_step2a_", "step2a_"])
    step2b = _find_model_file(md, ["cbm_step2b_", "step2b_"])
    step2c = _find_model_file(md, ["cbm_step2c_", "step2c_"])
    stepf = _find_model_file(md, ["cbm_stepf_", "stepf_"])

    missing = [k for k, v in {"step1": step1, "step2a": step2a, "step2b": step2b, "step2c": step2c, "stepf": stepf}.items() if not v]
    if missing:
        raise RuntimeError(f"model_dir에서 cbm 파일을 못 찾았습니다: {missing} (dir={md})")

    thresholds_json = _ensure_thresholds_json(md, training_log)

    # ✅ ModelRuntime 호환: stepF / stepf, step2A / step2a 등 둘 다 제공
    return SimpleNamespace(
        step1=step1,
        step2a=step2a,
        step2b=step2b,
        step2c=step2c,
        stepf=stepf,
        step2A=step2a,
        step2B=step2b,
        step2C=step2c,
        stepF=stepf,
        thresholds_json=thresholds_json,
    )


def run_predict(rt: ModelRuntime, bill_for_model: dict) -> dict:
    if hasattr(rt, "predict_8class_one"):
        return rt.predict_8class_one(bill_for_model)
    if hasattr(rt, "predict_one"):
        return rt.predict_one(bill_for_model)
    if hasattr(rt, "predict"):
        return rt.predict(bill_for_model)
    raise RuntimeError("ModelRuntime에서 1건 예측 함수(predict_8class_one / predict_one / predict)를 찾지 못했습니다.")


def _raw_to_X_one(raw: Dict[str, Any]) -> pd.DataFrame:
    """Build a single-row DataFrame for report_insights.

    DB rows may contain non-scalar types (pgvector, JSON/list, etc.).
    For report evidence we primarily need scalar meta features, so we drop
    non-scalar values to avoid CatBoost/Pool errors.
    """
    clean: Dict[str, Any] = {}
    for k, v in (raw or {}).items():
        if v is None or isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            # drop complex values (list/dict/bytes/objects)
            continue
    return pd.DataFrame([clean])


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _make_key_deltas(base: Dict[str, Any], other: Dict[str, Any]) -> str:
    num_keys = [
        "elapsed_days",
        "committee_meeting_count",
        "committee_meeting_days",
        "committee_referral_days",
        # ✅ 법사위 회의수 비교는 제거 (유사사례 핵심 차이에 안 뜨게)
        # "lwcmt_meeting_count",
        "proposer_count_est",
        "n_negotiation_groups",
    ]
    cat_keys = [
        "amendment_type",
        "proposer_kind",
        "party_at_term",
        "proc_stage_cd",
        "curr_committee_name",
        "is_alternative",
    ]

    parts: List[str] = []

    for k in num_keys:
        a = _safe_float(base.get(k))
        b = _safe_float(other.get(k))
        if a is None or b is None:
            continue
        diff = b - a
        if abs(diff) < 1e-9:
            continue
        sign = "+" if diff > 0 else ""
        if abs(diff - round(diff)) < 1e-9:
            diff_s = f"{sign}{int(round(diff))}"
        else:
            diff_s = f"{sign}{diff:.2f}"
        parts.append(f"{k} {diff_s}")

    for k in cat_keys:
        a = base.get(k)
        b = other.get(k)
        if a in (None, "") or b in (None, ""):
            continue
        if a != b:
            parts.append(f"{k}: {a}->{b}")

    if not parts:
        return "-"
    return "; ".join(parts[:4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db_url", required=True)
    ap.add_argument("--table", default="public.final_training_data_with_embedding")
    ap.add_argument("--bill_id", required=True)
    ap.add_argument("--out", default="sample_input.json")

    ap.add_argument("--training_log", default=None)
    ap.add_argument("--model_dir", default="backend/models")

    # similar cases options
    ap.add_argument("--sim_table", default=None, help="유사검색에 사용할 테이블(기본=--table)")
    ap.add_argument("--similar_top_k", type=int, default=10)
    ap.add_argument("--similar_same_committee", action="store_true", help="같은 curr_committee_name만 필터")

    args = ap.parse_args()

    engine = create_engine(args.db_url, pool_pre_ping=True)

    # 1) raw fetch
    raw = fetch_bill_raw(engine, args.table, args.bill_id)

    # 2) report용 정규화 bill
    bill = normalize_bill(raw)

    # 3) model predict (target)
    paths = build_paths(args.model_dir, args.training_log)
    rt = ModelRuntime(paths)
    pred = run_predict(rt, raw)

    # 4) training log parse (optional)
    model_info: Dict[str, Any] = {}
    if args.training_log and os.path.exists(args.training_log):
        log_text = Path(args.training_log).read_text(encoding="utf-8", errors="ignore")
        model_info["training_log_parsed"] = parse_training_log(log_text)

    # 5) similar cases (pgvector)
    sim_table = args.sim_table or args.table
    similar_cases = fetch_similar_cases(
        engine=engine,
        table_full=sim_table,
        bill_id=str(args.bill_id),
        top_k=int(args.similar_top_k),
        same_committee=bool(args.similar_same_committee),
        include_raw=True,
    )

    # rows 채우기(실제결과, key_deltas)
    for r in similar_cases.get("rows", []):
        other_raw = r.pop("_raw", None)
        if not isinstance(other_raw, dict):
            continue

        r["y_raw"] = other_raw.get("general_result", r.get("y_raw", "-"))
        r["key_deltas"] = _make_key_deltas(raw, other_raw)
        r["bill_name"] = r.get("bill_name") or other_raw.get("bill_name") or "-"
        # keep a short summary for content-level comparison in report (section 4)
        if not r.get("summary"):
            s = other_raw.get("summary")
            if s is not None:
                s = str(s)
                if len(s) > 2000:
                    s = s[:1999] + "…"
                r["summary"] = s

    # 6) evidence/insights (CatBoost SHAP/importance)
    evidence = None
    try:
        X_one = _raw_to_X_one(raw)
        evidence = build_report_insights(
            rt=rt,
            X_one=X_one,
            pred=pred,
            thresholds=pred.get("thresholds") if isinstance(pred.get("thresholds"), dict) else None,
            similar_cases=similar_cases,
            bill=bill,
            top_k_reasons=3,
            top_k_features=12,
        )
        if isinstance(pred, dict):
            pred["insights"] = evidence
    except Exception as e:
        print(f"// ⚠️ build_report_insights failed: {e}")
        evidence = None

    payload = {
        "bill": bill,
        "pred": pred,
        "evidence": evidence,
        "similar_cases": similar_cases,
        "timeline": [],
        "model_info": model_info,
        "db_url": args.db_url,
        "source_table": args.table,
        "sim_table": sim_table,
        "bill_raw": raw,
    }

    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"// ✅ wrote: {args.out}")
    print(f"// ✅ model_dir: {Path(args.model_dir).resolve()}")
    print(f"// ✅ table: {args.table}")
    print(f"//nd// ✅ sim_table: {sim_table}")


if __name__ == "__main__":
    main()
