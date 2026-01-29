# backend/report_builder.py
from __future__ import annotations

"""LLM enrich 결과를 포함한 '법안 처리결과 예측 보고서' 마크다운 빌더.

- LLM은 backend.llm_report_enricher.enrich_for_report(...)에서 섹션별로 보조 문장(요약/권고/해석)을 추가합니다.
- 이 파일은 (1) 원본 사실/구조를 유지하면서, (2) LLM이 만든 보조 문장을 사람이 읽기 좋은 형태로 렌더링합니다.

주의
- 모델/확률/임계값 숫자 노출은 최소화(사용자 관점 문장 위주)
"""

from typing import Any, Dict, List, Optional
import math
import re
from datetime import datetime


# -------------------------
# label sets
# -------------------------

PASS_SET = {"원안가결", "수정가결", "대안반영폐기", "수정안반영폐기"}
FAIL_SET = {"부결", "폐기", "철회", "임기만료폐기"}


# -------------------------
# feature/filter/method name mapping (영문/변수명 -> 한글)
# -------------------------

FEATURE_NAME_KO: Dict[str, str] = {
    # bill
    "bill_id": "법안 ID",
    "billId": "법안 ID",
    "bill_no": "법안번호",
    "billNo": "법안번호",
    "bill_name": "법안명",
    "bill_nm": "법안명",
    "propose_dt": "제안일",
    "ppsl_dt": "제안일",
    "proc_dt": "처리일",
    "elapsed_days": "발의 후 경과 기간",
    "proposer_count_est": "발의자 수",
    "proposer_kind": "발의자 구분",
    "proposer_name": "발의자명",
    "ppsr_nm": "발의자명",
    "pass_gubn": "법안 구분",
    "proc_stage_cd": "현재 상태",
    "general_result": "처리 결과",
    "budget": "비용추계서 유무",
    "budget_yn": "비용추계서 유무",
    "curr_committee_code": "소관위원회 코드",
    "curr_committee_name": "소관위원회명",
    "committee_name": "소관위원회명",
    "competent_ministry": "소관부처",
    "committee_referral_count": "소관위 회부 수",
    "committee_referral_days": "소관위 회부 소요기간",
    "committee_meeting_count": "소관위 회의 수",
    "committee_meeting_days": "소관위 회의 소요기간",
    "is_alternative": "대안 여부",
    "alternative_yn": "대안 여부",
    "amendment_type": "제개정 구분",
    "lwcmt_meeting_count": "법사위 회의수",
    "summary": "법안 개요",
    "ord": "대수",
    # 의원
    "naas_cd": "의원 ID",
    "naas_nm": "의원명",
    "term": "대수",
    "reelection_count": "재선 횟수",
    "ntr_div": "성별",
    "election_type": "비례/지역구",
    "district_at_term": "선거구",
    "party_at_term": "당시 정당",
    "party_seats": "정당 의석 수",
    "is_negotiation_group_party": "교섭단체 여부",
    "has_committee": "위원회 소속 여부",
    "committee_at_term": "상임위원회 소속 여부",
    "n_negotiation_groups": "교섭단체 수",
    # report-only / extra
    "log_elapsed": "발의 후 경과 기간(로그)",
    "propose_year": "제안년도",
    "ai_prediction": "AI 예측",
    "ai_probability": "AI 확률",
    "ai_report": "AI 보고서",
    "expert_report": "전문가 보고서",
    "social_issue_yn": "사회적 이슈 여부",
    "social_issue_score": "사회적 이슈 점수",
}

FILTER_NAME_KO: Dict[str, str] = {
    "same_committee": "발의자 소관위 소속 여부",
    "curr_committee_name": "소관위원회명",
    "curr_committee_code": "소관위원회 코드",
}

METHOD_NAME_KO: Dict[str, str] = {
    "pgvector cosine": "코사인",
}


def _feature_ko(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return "-"
    return FEATURE_NAME_KO.get(n, n)


def _filter_ko(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return "-"
    return FILTER_NAME_KO.get(n, n)


def _method_ko(text: str) -> str:
    if not text:
        return "-"
    out = text
    for k, v in METHOD_NAME_KO.items():
        out = out.replace(k, v)
    return out


def _safe(x: Any, default: str = "-") -> str:
    if x is None:
        return default
    s = str(x).strip()
    return s if s else default


def _fmt_prob(x: Any, nd: int = 4) -> str:
    try:
        v = float(x)
        if math.isnan(v):
            return "-"
        return f"{v:.{nd}f}"
    except Exception:
        return "-"


def _fmt_int_commas(x: Any) -> str:
    try:
        if isinstance(x, bool):
            return str(x)
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            if isinstance(x, float) and math.isnan(x):
                return "-"
            iv = int(round(float(x)))
            return f"{iv:,}"
        s = str(x).strip()
        if re.fullmatch(r"-?\d+", s):
            return f"{int(s):,}"
        if re.fullmatch(r"-?\d+\.\d+", s):
            iv = int(round(float(s)))
            return f"{iv:,}"
        return s
    except Exception:
        return _safe(x)


def _fmt_value(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, str) and not x.strip():
        return "-"
    return _fmt_int_commas(x)


def _replace_feature_tokens_in_text(text: str) -> str:
    """desc/해석 문자열 안에 raw 변수명이 섞여 있으면 한글명으로 치환."""
    if not text:
        return "-"
    out = text
    for k in sorted(FEATURE_NAME_KO.keys(), key=len, reverse=True):
        if k and k in out:
            out = out.replace(k, FEATURE_NAME_KO[k])
    for k in sorted(FILTER_NAME_KO.keys(), key=len, reverse=True):
        if k and k in out:
            out = out.replace(k, FILTER_NAME_KO[k])
    for k, v in METHOD_NAME_KO.items():
        if k and k in out:
            out = out.replace(k, v)
    return out


# -------------------------
# thresholds normalizer (구버전/신버전 키 혼재 대응)
# -------------------------

def _normalize_thresholds(pred: Dict[str, Any]) -> Dict[str, Any]:
    th = pred.get("thresholds") or {}
    if not isinstance(th, dict):
        th = {}

    out = dict(th)

    if "fail" not in out:
        out["fail"] = th.get("thr_fail_step1", th.get("thr_fail"))
    if "non_alt" not in out:
        out["non_alt"] = th.get("thr_non_alt_step2a", th.get("thr_non_alt"))
    if "orig" not in out:
        out["orig"] = th.get("thr_orig_step2b", th.get("thr_orig"))
    if "alt" not in out:
        out["alt"] = th.get("thr_alt_step2c", th.get("thr_alt"))

    return out


# -------------------------
# confidence (휴리스틱) + topk(근사) 복구
# -------------------------

def _confidence_level(pred: Dict[str, Any]) -> str:
    ins = pred.get("insights")
    if isinstance(ins, dict):
        c = str(ins.get("confidence") or "").strip()
        if c:
            return c

    th = _normalize_thresholds(pred)
    p_fail = pred.get("p_fail_step1")
    p_non_alt = pred.get("p_non_alt")
    p_orig = pred.get("p_orig")
    p_alt = pred.get("p_alt")

    thr_fail = th.get("fail")
    thr_non_alt = th.get("non_alt")
    thr_orig = th.get("orig")
    thr_alt = th.get("alt")

    def _margin(p: Any, t: Any) -> Optional[float]:
        try:
            if p is None or t is None:
                return None
            return abs(float(p) - float(t))
        except Exception:
            return None

    margins = [
        _margin(p_fail, thr_fail),
        _margin(p_non_alt, thr_non_alt),
        _margin(p_orig, thr_orig),
        _margin(p_alt, thr_alt),
    ]
    margins = [m for m in margins if m is not None]
    if not margins:
        return "Medium"

    m = min(margins)
    if m >= 0.20:
        return "High"
    if m >= 0.10:
        return "Medium"
    return "Low"


def _approx_topk(pred: Dict[str, Any], k: int = 3) -> List[Dict[str, Any]]:
    p_fail = pred.get("p_fail_step1")
    p_non_alt = pred.get("p_non_alt")
    p_orig = pred.get("p_orig")
    p_alt = pred.get("p_alt")
    final_label = str(pred.get("final_pred_8") or pred.get("final_pred") or "")

    fail_topk = pred.get("fail_topk") or pred.get("fail_proba_topk")

    scores: Dict[str, float] = {lbl: 0.0 for lbl in (PASS_SET | FAIL_SET)}

    def _to_float(x: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            if x is None:
                return default
            v = float(x)
            if math.isnan(v):
                return default
            return v
        except Exception:
            return default

    pf = _to_float(p_fail, None)
    if pf is None:
        if final_label in scores:
            scores[final_label] = 1.0
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [{"label": a, "prob": float(b)} for a, b in top]

    p_pass = max(0.0, 1.0 - pf)
    pna = _to_float(p_non_alt, 0.5) or 0.5
    por = _to_float(p_orig, 0.5) or 0.5
    pal = _to_float(p_alt, 0.5) or 0.5

    scores["원안가결"] = p_pass * pna * por
    scores["수정가결"] = p_pass * pna * (1.0 - por)
    scores["수정안반영폐기"] = p_pass * (1.0 - pna) * pal
    scores["대안반영폐기"] = p_pass * (1.0 - pna) * (1.0 - pal)

    if isinstance(fail_topk, list) and fail_topk:
        for item in fail_topk:
            try:
                lbl, pr = item
                if lbl in scores:
                    scores[lbl] = max(scores[lbl], pf * float(pr))
            except Exception:
                continue
    else:
        if final_label in FAIL_SET:
            scores[final_label] = max(scores[final_label], float(pf))

    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [{"label": a, "prob": float(b)} for a, b in top]


# -------------------------
# ✅ 변경: Section 2.2 decision path를 '단계별 확률 요약'으로 간결화
# -------------------------

def _fmt_pct01(x: Any, nd: int = 0) -> str:
    try:
        if x is None:
            return "-"
        v = float(x)
        if math.isnan(v):
            return "-"
        v = max(0.0, min(1.0, v))
        return f"{v * 100:.{nd}f}%"
    except Exception:
        return "-"


def _decision_sentences(pred: Dict[str, Any]) -> List[str]:
    p_fail = pred.get("p_fail_step1")
    p_non_alt = pred.get("p_non_alt")
    p_orig = pred.get("p_orig")
    p_alt = pred.get("p_alt")
    thr = _normalize_thresholds(pred)

    thr_fail = thr.get("fail")
    thr_non_alt = thr.get("non_alt")
    thr_orig = thr.get("orig")
    thr_alt = thr.get("alt")

    sents: List[str] = []

    def _lt(a: Any, b: Any) -> Optional[bool]:
        try:
            if a is None or b is None:
                return None
            return float(a) < float(b)
        except Exception:
            return None

    def _ge(a: Any, b: Any) -> Optional[bool]:
        try:
            if a is None or b is None:
                return None
            return float(a) >= float(b)
        except Exception:
            return None

    if p_fail is None:
        sents.append("- **1단계(통과/비통과):** (신호 부족) → 분기 해석 제한")
        return sents

    try:
        pf = float(p_fail)
        if math.isnan(pf):
            pf = None  # type: ignore
    except Exception:
        pf = None  # type: ignore

    if pf is None:
        sents.append("- **1단계(통과/비통과):** (신호 부족) → 분기 해석 제한")
        return sents

    p_pass = 1.0 - pf
    pass_like = _lt(p_fail, thr_fail)
    chosen1 = "통과" if pass_like is True else ("비통과" if pass_like is False else "종합")
    sents.append(f"- **1단계(통과/비통과):** 통과 {_fmt_pct01(p_pass)} · 비통과 {_fmt_pct01(pf)} → **{chosen1}**")

    if pass_like is False:
        return sents

    non_alt_like = _ge(p_non_alt, thr_non_alt)
    if p_non_alt is None:
        sents.append("- **2단계(비대안/대안):** (신호 부족) → 세부 분기 해석 제한")
        return sents

    try:
        pna = float(p_non_alt)
        if math.isnan(pna):
            raise ValueError()
    except Exception:
        sents.append("- **2단계(비대안/대안):** (신호 부족) → 세부 분기 해석 제한")
        return sents

    pa = 1.0 - pna
    chosen2 = "비대안" if non_alt_like is True else ("대안" if non_alt_like is False else "종합")
    sents.append(f"- **2단계(비대안/대안):** 비대안 {_fmt_pct01(pna)} · 대안 {_fmt_pct01(pa)} → **{chosen2}**")

    if non_alt_like is True:
        if p_orig is None:
            sents.append("- **3단계(원안/수정):** (신호 부족) → 세부 분기 해석 제한")
            return sents
        try:
            por = float(p_orig)
            if math.isnan(por):
                raise ValueError()
        except Exception:
            sents.append("- **3단계(원안/수정):** (신호 부족) → 세부 분기 해석 제한")
            return sents

        pam = 1.0 - por
        orig_like = _ge(p_orig, thr_orig)
        chosen3 = "원안" if orig_like is True else ("수정" if orig_like is False else "종합")
        sents.append(f"- **3단계(원안/수정):** 원안 {_fmt_pct01(por)} · 수정 {_fmt_pct01(pam)} → **{chosen3}**")
        return sents

    if non_alt_like is False:
        if p_alt is None:
            sents.append("- **3단계(대안 처리 내 분기):** (신호 부족) → 세부 분기 해석 제한")
            return sents
        try:
            pal = float(p_alt)
            if math.isnan(pal):
                raise ValueError()
        except Exception:
            sents.append("- **3단계(대안 처리 내 분기):** (신호 부족) → 세부 분기 해석 제한")
            return sents

        palt2 = 1.0 - pal
        alt_like = _ge(p_alt, thr_alt)
        chosen3 = "수정안반영폐기" if alt_like is True else ("대안반영폐기" if alt_like is False else "종합")
        sents.append(
            f"- **3단계(대안 처리 내 분기):** 수정안반영폐기 {_fmt_pct01(pal)} · 대안반영폐기 {_fmt_pct01(palt2)} → **{chosen3}**"
        )
        return sents

    sents.append("- **3단계:** (신호 부족) → 세부 분기 해석 제한")
    return sents


# -------------------------
# Similar cases: delta humanizer
# -------------------------

_RE_DELTA_PLUS = re.compile(r"^\s*([A-Za-z0-9_]+)\s*([+-])\s*([0-9,]+)\s*$")
_RE_DELTA_ARROW = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*(.+?)\s*(?:->|→)\s*(.+?)\s*$")
_RE_DELTA_KV = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*(.+)\s*$")


def _drop_elapsed_related(delta_key: str) -> bool:
    k = (delta_key or "").strip().lower()
    return k in {"elapsed_days", "elapsed", "경과일", "발의 후 경과 기간"} or "elapsed" in k or "경과" in k


def _humanize_delta_item(item: str) -> Optional[str]:
    if not item:
        return None
    s = item.strip()
    if not s or s == "-":
        return None

    m = _RE_DELTA_PLUS.match(s)
    if m:
        key, sign, num = m.group(1), m.group(2), m.group(3)
        if _drop_elapsed_related(key):
            return None
        return f"{_feature_ko(key)} {sign}{num}"

    m = _RE_DELTA_ARROW.match(s)
    if m:
        key, a, b = m.group(1), m.group(2).strip(), m.group(3).strip()
        if _drop_elapsed_related(key):
            return None
        return f"{_feature_ko(key)}: {a}→{b}"

    m = _RE_DELTA_KV.match(s)
    if m:
        key, rest = m.group(1), m.group(2).strip()
        if _drop_elapsed_related(key):
            return None
        if "elapsed" in rest.lower() or "경과" in rest:
            return None
        return f"{_feature_ko(key)}: {rest}"

    if "elapsed" in s.lower() or "경과" in s:
        return None
    return s


def _humanize_deltas(deltas: Any) -> str:
    if deltas is None:
        return "-"
    if isinstance(deltas, list):
        parts: List[str] = []
        for x in deltas:
            hx = _humanize_delta_item(str(x))
            if hx:
                parts.append(hx)
        return "; ".join(parts) if parts else "-"

    if isinstance(deltas, str):
        raw = deltas.strip()
        if not raw or raw == "-":
            return "-"
        chunks = re.split(r"\s*;\s*|\s*,\s*|\s*\|\s*", raw)
        parts = []
        for c in chunks:
            hx = _humanize_delta_item(c)
            if hx:
                parts.append(hx)
        return "; ".join(parts) if parts else "-"

    return _safe(deltas)


# -------------------------
# Renderers
# -------------------------

def _render_header(now: Optional[str] = None) -> str:
    ts = now or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"# 법안 처리결과 예측 보고서\n\n- 생성시각: `{ts}`\n"


def _render_section0(pred: Dict[str, Any], evidence: Dict[str, Any]) -> str:
    final_pred = _safe(pred.get("final_pred_8") or pred.get("final_pred"), "-")
    group = "PASS" if final_pred in PASS_SET else ("FAIL" if final_pred in FAIL_SET else "-")

    conf = _safe((pred.get("insights") or {}).get("confidence"), "")
    if not conf or conf == "-":
        conf = _confidence_level(pred)

    top3 = evidence.get("top_reasons")
    lines: List[str] = []
    if isinstance(top3, list):
        for i, x in enumerate(top3[:3], start=1):
            s = _safe(x, "-")
            lines.append(f"  {i}. {s}")
    if not lines:
        feats = evidence.get("top_features") if isinstance(evidence, dict) else None
        if isinstance(feats, list):
            for i, f in enumerate(feats[:3], start=1):
                if isinstance(f, dict):
                    desc = _replace_feature_tokens_in_text(_safe(f.get("desc") or f.get("interpretation"), "-"))
                    lines.append(f"  {i}. {desc}")
    if not lines:
        lines = [
            "  1. (근거 데이터가 부족하여 자동 생성이 제한됩니다)",
            "  2. (근거 데이터가 부족하여 자동 생성이 제한됩니다)",
            "  3. (근거 데이터가 부족하여 자동 생성이 제한됩니다)",
        ]

    return (
        "## 0. 요약\n\n"
        f"- **최종 분류:** `{final_pred}`\n"
        f"- **신뢰도:** **{conf}**\n"
        "- **핵심 근거:**\n"
        + "\n".join(lines)
        + "\n"
    )


def _get_top_feature_value(evidence: Dict[str, Any], *, raw_names: List[str], name_contains: Optional[str] = None) -> Optional[Any]:
    feats = evidence.get("top_features")
    if not isinstance(feats, list):
        return None
    raw_set = {x.strip().lower() for x in raw_names if x}
    for f in feats:
        if not isinstance(f, dict):
            continue
        raw = str(f.get("raw_name") or f.get("name") or "").strip()
        nm = str(f.get("name") or "").strip()
        raw_low = raw.lower()
        if raw_low in raw_set:
            return f.get("value")
        if name_contains and name_contains in nm:
            return f.get("value")
    return None


def _infer_ord_from_bill_no(bill_no: str) -> Optional[int]:
    s = re.sub(r"\D", "", (bill_no or ""))
    if len(s) < 2:
        return None
    try:
        v = int(s[:2])
        if 1 <= v <= 99:
            return v
    except Exception:
        return None
    return None


def _render_section1(bill: Dict[str, Any], evidence: Dict[str, Any]) -> str:
    bill_id = _safe(bill.get("bill_id") or bill.get("billId"))
    bill_no = _safe(bill.get("bill_no") or bill.get("billNo"))
    bill_nm = _safe(bill.get("bill_name") or bill.get("bill_nm"))
    propose_dt = _safe(bill.get("propose_dt") or bill.get("ppsl_dt"))
    committee = _safe(bill.get("curr_committee_name") or bill.get("committee_name"))
    proposer = _safe(bill.get("proposer_name") or bill.get("ppsr_nm"))
    proposer_cnt = _safe(bill.get("proposer_count_est"))
    amend = _safe(bill.get("amendment_type"))


    # ---- helpers (섹션1 표시용 fallback) ----
    def _is_missing(v: Any) -> bool:
        if v is None:
            return True
        s = str(v).strip()
        return (not s) or (s.lower() in {"-", "none", "null", "nan"})

    def _from_top_features(*keys: str) -> Optional[Any]:
        feats = evidence.get("top_features")
        if not isinstance(feats, list):
            return None
        keyset = {k.strip() for k in keys if k and k.strip()}
        if not keyset:
            return None
        for f in feats:
            if not isinstance(f, dict):
                continue
            raw = str(f.get("raw_name") or "").strip()
            name = str(f.get("name") or "").strip()
            if raw in keyset or name in keyset:
                v = f.get("value")
                if not _is_missing(v):
                    return v
        return None

    # ✅ (요청) 섹션 1에 '대수' + '의원 정당 소속(party_at_term)' 표시
    # 대수: ord/term 외에 DB에서 많이 쓰는 eraco도 fallback
    ord_val = bill.get("ord")
    print(ord_val)
    if _is_missing(ord_val):
        ord_val = bill.get("eraco")  # ✅ 추가 fallback
    if _is_missing(ord_val):
        ord_val = bill.get("term")
    if _is_missing(ord_val):
        ord_val = _from_top_features("ord", "eraco", "term", "대수")

    ord_str = "-" if _is_missing(ord_val) else _fmt_value(ord_val)

    party_val = bill.get("party_at_term")
    if _is_missing(party_val):
        party_val = bill.get("party")
    if _is_missing(party_val):
        party_val = _from_top_features(
            "party_at_term",
            "당시 정당",
            "당(대수 기준)",
            "정당",
            "발의자 소속 정당",
        )

    party_str = "-" if _is_missing(party_val) else _safe(party_val, "-")

    lines = evidence.get("bill_summary_lines")
    summary_block = ""
    if isinstance(lines, list) and len(lines) >= 3:
        summary_block = "- **요약:**\n" + "\n".join([f"  - {_safe(x)}" for x in lines[:5]]) + "\n"
    else:
        summ = _safe(bill.get("summary"), "-")
        summary_block = f"- **요약(summary):** {summ}\n"

    return (
        "## 1. 입력 법안 정보\n\n"
        f"- **법안번호:** {bill_no}\n"
        f"- **법안명:** {bill_nm}\n"
        f"- **발의일:** {propose_dt}\n"
        f"- **소관위:** {committee}\n"
        f"- **발의자:** {proposer}\n"
        f"- **발의자 수:** {proposer_cnt}\n"
        f"- **발의자 소속 정당:** {party_str}\n"
        f"- **제개정 구분:** {amend}\n"
        f"{summary_block}"
    )


def _render_section2(pred: Dict[str, Any]) -> str:
    topk = pred.get("topk")
    if not isinstance(topk, list) or not topk:
        topk = _approx_topk(pred, k=3)

    rows: List[str] = []
    if isinstance(topk, list):
        for i, x in enumerate(topk[:3], start=1):
            if isinstance(x, dict):
                label = _safe(x.get("label"), "-")
                prob = _fmt_prob(x.get("prob"))
                rows.append(f"| {i} | {label} | {prob} |")
            elif isinstance(x, (list, tuple)) and len(x) >= 2:
                label = _safe(x[0], "-")
                prob = _fmt_prob(x[1])
                rows.append(f"| {i} | {label} | {prob} |")

    dist_table = ""
    if rows:
        dist_table = (
            "| 순위 | 예측 | 확률 |\n"
            "| --- | --- | --- |\n"
            + "\n".join(rows)
            + "\n\n"
        )

    return (
        "## 2. 모델 예측 결과\n\n"
        + dist_table
    )


def _render_section3(evidence: Dict[str, Any]) -> str:
    feats = evidence.get("top_features")
    table_rows: List[str] = []

    if isinstance(feats, list):
        for i, f in enumerate(feats[:12], start=1):
            if not isinstance(f, dict):
                continue
            raw_name = _safe(f.get("raw_name") or f.get("name"), "-")
            disp_name = _safe(f.get("name"), "")
            feature_name = _feature_ko(disp_name) if disp_name and disp_name != "-" else _feature_ko(raw_name)

            value_raw = f.get("value")
            raw_lower = (raw_name or "").strip().lower()
            if feature_name == "법안 개요" or raw_lower == "summary":
                value = "-" if value_raw is None or (isinstance(value_raw, str) and not value_raw.strip()) else "제안 이유 및 주요 내용"
            else:
                value = _fmt_value(value_raw)

            desc = _replace_feature_tokens_in_text(_safe(f.get("desc") or f.get("interpretation"), "-"))

            table_rows.append(f"| {i} | {feature_name} | {value} | {desc} |")

    process_summary = evidence.get("process_summary")
    ps = ""
    if process_summary:
        ps = f"\n- **해석 요약:** {str(process_summary).strip()}\n"

    if table_rows:
        table = (
            "| 순위 | 피처 | 값 | 해석 |\n"
            "| --- | --- | --- | --- |\n"
            + "\n".join(table_rows)
            + "\n"
        )
    else:
        table = "(피처 근거 데이터가 부족하여 표를 생성할 수 없습니다.)\n"

    return "## 3. 근거 분석\n\n" + table + ps


def _render_section4(similar_cases: Dict[str, Any]) -> str:
    lines = (
        "## 4. 유사 사례 비교분석\n\n"
        "\n"
    )

    rows = similar_cases.get("rows")
    if isinstance(rows, list) and rows:
        trows = []
        for i, r in enumerate(rows[:10], start=1):
            if not isinstance(r, dict):
                continue
            bill_nm = _safe(r.get("bill_nm") or r.get("bill_name"), "-")
            y_raw = _safe(r.get("y_raw"), "-")
            sim = _fmt_prob(r.get("similarity") or r.get("score"))

            # ✅ key_deltas_text(LLM 서술형)이 있으면 우선 사용
            deltas_text = r.get("key_deltas_text") or r.get("delta_text")
            if isinstance(deltas_text, str) and deltas_text.strip():
                deltas = _replace_feature_tokens_in_text(deltas_text.strip())
            else:
                deltas = _humanize_deltas(r.get("key_deltas") or r.get("delta"))

            trows.append(f"| {i} | {bill_nm} | {y_raw} |")

        lines += (
            "| 순위 | 법안명 | 처리 결과 |\n"
            "| --- | --- | --- | --- |\n"
            + "\n".join(trows)
            + "\n"
        )
    else:
        lines += "(유사사례 데이터가 없습니다.)\n"

    hints = similar_cases.get("summary_hints")
    if isinstance(hints, dict):
        common = hints.get("common_patterns")
        diff = hints.get("diff_patterns")
        conclusion = _safe(hints.get("conclusion"), "-")

        common_lines = ""
        if isinstance(common, list) and common:
            common_lines = "\n".join([f"  - {_replace_feature_tokens_in_text(_safe(x))}" for x in common[:3]])
        diff_lines = ""
        if isinstance(diff, list) and diff:
            diff_lines = "\n".join([f"  - {_replace_feature_tokens_in_text(_safe(x))}" for x in diff[:3]])

        if common_lines or diff_lines or conclusion != "-":
            lines += (
                "\n### 4.1 유사사례 결과·차이 기반 해석\n"
                "\n"
                "- **유사 법안들의 실제 결과/맥락에서 자주 보인 패턴**\n"
                + (common_lines + "\n" if common_lines else "  - (자동 생성 근거가 부족합니다)\n")
                + "- **이번 법안의 핵심 차이로 본 점검 포인트**\n"
                + (diff_lines + "\n" if diff_lines else "  - (자동 생성 근거가 부족합니다)\n")
                + f"- **종합:** {_replace_feature_tokens_in_text(conclusion)}\n"
            )

    return lines


# -------------------------
# ✅ NEW: Section 5. 사회적 영향력 분석
# -------------------------

def _render_section5_social(evidence: Dict[str, Any], bill: Dict[str, Any]) -> str:
    """
    evidence에서 아래 키를 기대:
    - social_impact_score: 0~100
    - social_impact_lines: 3줄(리스트)
    - social_impact_sources: 2~5개(리스트)
    """
    score = evidence.get("social_impact_score")
    lines = evidence.get("social_impact_lines")
    sources = evidence.get("social_impact_sources")

    # score formatting
    score_str = "-"
    try:
        if score is None or (isinstance(score, str) and not score.strip()):
            raise ValueError()
        v = float(score)
        if math.isnan(v):
            raise ValueError()
        v = max(0.0, min(100.0, v))
        # 0.0~100.0 연속값이지만, 보고서엔 가독성 있게 0~100 정수 우선
        score_str = f"{int(round(v))}"
    except Exception:
        score_str = "-"

    rendered_lines: List[str] = []
    if isinstance(lines, list):
        for x in lines[:3]:
            s = _safe(x, "").strip()
            if s:
                rendered_lines.append(s)

    if len(rendered_lines) < 3:
        rendered_lines = [
            "웹 근거가 충분하지 않아, 당시 사회적 이슈화 정도를 보수적으로 해석할 필요가 있습니다.",
            "해당 사안이 전국적 의제로 확산되었는지 여부는 추가 자료(언론 보도/공식 자료) 확인이 도움이 됩니다.",
            "현재 수집 가능한 출처 범위가 제한되어 점수는 참고용으로만 활용하는 것이 안전합니다.",
        ]

    src_line = ""
    if isinstance(sources, list) and sources:
        srcs = [str(x).strip() for x in sources[:5] if str(x).strip()]
        if srcs:
            src_line = f"- **참고 출처:** {', '.join(srcs)}\n"

    return (
        "## 5. 사회적 영향력 분석\n\n"
        f"- **사회적 영향력 점수(0~100):** {score_str}\n"
        "- **분석:**\n"
        + "\n".join([f"  - {_replace_feature_tokens_in_text(s)}" for s in rendered_lines[:3]])
        + "\n"
        + src_line
    )


# -------------------------
# ✅ 기존 Section 5를 Section 6으로 "그대로" 이동 (내용 동일, 번호만 변경)
# -------------------------

def _render_section6(evidence: Dict[str, Any], bill: Dict[str, Any]) -> str:
    nxt = evidence.get("next_actions")

    if isinstance(nxt, list) and nxt:
        rendered_blocks: List[str] = []
        for x in nxt[:3]:
            s = str(x).rstrip()
            if not s:
                continue
            s = _replace_feature_tokens_in_text(s)
            if s.lstrip().startswith("- "):
                rendered_blocks.append(s)
            else:
                rendered_blocks.append(f"- {s}")
        if rendered_blocks:
            return "## 6. 시사점\n\n" + "\n\n".join(rendered_blocks) + "\n"

    return (
        "## 6. 시사점\n\n"
        "- 법안 주요 변경점이 여러 항목에 걸쳐 있는 만큼, 각 조문에서 **적용 시점·경과규정·예외 규정**을 함께 정리해볼 필요가 있습니다.\n"
        "  - 동일 개념의 용어/정의가 조문마다 달라지지 않는지 확인해볼 수 있습니다.\n"
        "  - 적용 대상(범위)과 제외 대상(예외)을 분리해 문장 구조를 명확히 하는 것이 도움이 될 수 있습니다.\n"
        "  (근거: 집행 단계에서 해석 차이로 혼선이 생기는 경우가 있습니다)\n"
        "- 이해관계자(대상자·사업주·행정기관) 부담이 바뀌는 조항은, **증빙·절차·제재 흐름**을 함께 점검해볼 필요가 있습니다.\n"
        "  - 제출/확인 절차가 현실적으로 가능한지 확인해볼 수 있습니다.\n"
        "  - 제재 규정은 정의(요건)와 절차(통지·이의제기)까지 함께 정리해볼 수 있습니다.\n"
        "  (근거: 유사 제재 입법에서 절차 명확성이 쟁점이 되는 경우가 있습니다)\n"
        "- 유사사례에서 처리결과가 엇갈린 경우가 있다면, 본 법안의 **발의 주체·절차 흐름·정치적 맥락** 차이를 비교해볼 수 있습니다.\n"
        "  - 유사하더라도 처리 경로가 다른 경우 결과가 달라질 수 있습니다.\n"
        "  - 소관위/법사위 단계의 논의 여부를 함께 확인해볼 수 있습니다.\n"
        "  (근거: 텍스트 유사도는 절차/맥락 차이를 완전히 반영하지 못할 수 있습니다)\n"
    )


def build_report_markdown(
    *,
    bill: Dict[str, Any],
    pred: Dict[str, Any],
    evidence: Optional[Dict[str, Any]] = None,
    similar_cases: Optional[Dict[str, Any]] = None,
    timeline: Any = None,
    model_info: Optional[Dict[str, Any]] = None,
    now: Optional[str] = None,
) -> str:
    evidence = evidence or {}
    similar_cases = similar_cases or {}

    eff_evidence: Dict[str, Any] = {}
    if isinstance(pred.get("insights"), dict):
        eff_evidence.update(pred["insights"])
    eff_evidence.update(evidence)

    parts: List[str] = []
    parts.append(_render_header(now))
    parts.append("\n")
    parts.append(_render_section0(pred, eff_evidence))
    parts.append("\n")
    parts.append(_render_section1(bill, eff_evidence))
    parts.append("\n")
    parts.append(_render_section2(pred))
    parts.append("\n")
    parts.append(_render_section3(eff_evidence))
    parts.append("\n")
    if isinstance(similar_cases, dict):
        parts.append(_render_section4(similar_cases))
    else:
        parts.append("## 4. 유사 사례 비교분석\n\n(유사사례 데이터 형식이 올바르지 않습니다.)\n")
    parts.append("\n")

    # ✅ NEW 5
    parts.append(_render_section5_social(eff_evidence, bill))
    parts.append("\n")

    # ✅ OLD 5 -> 6 (내용/로직 그대로)
    parts.append(_render_section6(eff_evidence, bill))

    return "".join(parts)


__all__ = ["build_report_markdown"]
