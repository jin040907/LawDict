# backend/llm_report_enricher.py
from __future__ import annotations

"""LLM 기반 '문장화/요약' 보조 모듈.

핵심 설계
- 보고서의 '사실'과 '구조'는 코드(report_insights/report_builder)에서 확정합니다.
- LLM은 섹션별로 **문장화/요약/표현 개선**만 수행합니다.
- 섹션별 프롬프트를 분리하여(0/1/3/4/5) 오염(확률값을 피처 근거로 착각 등)을 방지합니다.

안전장치
- OPENAI_API_KEY가 없거나 호출 실패 시: 예외를 던지지 않고 그대로 반환(휴리스틱만으로도 보고서 생성).
- 출력은 Pydantic 스키마(JSON)로 강제.

사용 위치
- scripts/generate_sample_report_llm.py 에서 enrich_for_report(...) 호출
"""

import json
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore
    Field = lambda *a, **k: None  # type: ignore

from dotenv import load_dotenv
from pathlib import Path

# 환경 변수 로드: report/.env, service/.env, 프로젝트 루트 .env 순으로 시도
_here = Path(__file__).resolve().parent
for _p in [_here.parent / ".env", _here.parent.parent / "service" / ".env", _here.parent.parent / ".env"]:
    if _p.exists():
        load_dotenv(dotenv_path=_p, override=True)
        break
else:
    load_dotenv(override=True)

# -------------------------
# schemas (섹션별)
# -------------------------

class Sec0Out(BaseModel):
    top_reasons: List[str] = Field(..., description="0.요약에 들어갈 핵심 근거 3개(피처 기반).")


class Sec1Out(BaseModel):
    summary_lines: List[str] = Field(..., description="법안 요약 5줄(사용자용, 1줄 1문장).")


class Sec3Out(BaseModel):
    process_summary: str = Field(..., description="3.근거분석 아래에 들어갈 사용자 관점 요약(2~4문장).")


# ✅ 섹션 3 표 내부 '해석' 한줄씩 생성
class Sec3TableOut(BaseModel):
    row_interpretations: List[str] = Field(..., description="3.근거 분석 표의 각 행(순위|피처|값)에 대한 한 줄 해석(동일 순서).")


class Sec4Out(BaseModel):
    common_patterns: List[str] = Field(..., description="유사사례 결과/맥락에서 자주 보인 패턴 2~3개")
    diff_patterns: List[str] = Field(..., description="이번 법안의 핵심 차이로 본 점검 포인트 2~3개")
    similar_conclusion: str = Field(..., description="종합 결론 1~2문장(단정/과장 금지)")


# ✅ (요청 반영) 섹션 4 표 '핵심 차이'를 서술형으로 변환
class Sec4RowDeltaOut(BaseModel):
    row_delta_texts: List[str] = Field(..., description="유사사례 표의 각 행 핵심 차이를 서술형 1문장으로 변환(동일 순서).")


# ✅ NEW: 섹션 5(사회적 영향력 분석) 출력
class Sec5SocialOut(BaseModel):
    social_impact_score: float = Field(..., description="발의일 당시 사회 이슈화 점수(0~100 연속값).")
    social_impact_lines: List[str] = Field(..., description="사회적 영향력 분석 3줄(각 1문장).")
    sources: List[str] = Field(..., description="참고 출처 2~5개(입력 search_results에서만).")


class Sec5Out(BaseModel):
    # 각 항목은 '마크다운 블록' 한 덩어리(루트 bullet + 하위 bullet + 근거)
    next_actions_md: List[str] = Field(..., description="6.향후 대응 방향 3개(각 항목은 마크다운 bullet 블록).")


class _KwOut(BaseModel):
    keywords: List[str]


# -------------------------
# helpers
# -------------------------

MISSING_STRS = {"", "none", "null", "nan", "na", "n/a", "-", "없음"}

_RE_DELTA_PLUS = re.compile(r"^\s*([A-Za-z0-9_]+)\s*([+-])\s*([0-9,]+(?:\.\d+)?)\s*$")
_RE_DELTA_ARROW = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*(.+?)\s*(?:->|→)\s*(.+?)\s*$")
_RE_DELTA_KV = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*(.+)\s*$")

# 최소한의 키만 한글로 보강(표 '핵심 차이' 서술을 자연스럽게 만들기 위함)
_DELTA_KEY_KO: Dict[str, str] = {
    "proposer_count_est": "발의자 수",
    "proposer_kind": "발의자 구분",
    "party_at_term": "당시 정당",
    "is_alternative": "대안 여부",
    "alternative_yn": "대안 여부",
    "amendment_type": "개정유형",
    "curr_committee_name": "소관위원회명",
    "committee_name": "소관위원회명",
    "curr_committee_code": "소관위원회 코드",
    "n_negotiation_groups": "교섭단체 수",
    "election_type": "비례/지역구",
    "elapsed_days": "발의 후 경과 기간",
    "log_elapsed": "발의 후 경과 기간(로그)",
}


def _clip_text(x: Any, max_len: int = 1600) -> str:
    if x is None:
        return ""
    s = str(x)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _has_openai_key() -> bool:
    if os.getenv("OPENAI_API_KEY"):
        return True
    if os.getenv("OPENAI_KEY"):
        return True
    return False


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _banned_feature_name(name: str) -> bool:
    low = (name or "").strip().lower()
    if not low:
        return True
    banned_prefix = ("p_", "thr_", "threshold", "final_pred", "model_", "fail_topk")
    banned_exact = {"final_pred_8", "p_fail_step1", "p_non_alt", "p_orig", "p_alt"}
    if name in banned_exact:
        return True
    if any(low.startswith(p) for p in banned_prefix):
        return True
    return False


def _select_top_features(evidence: Optional[Dict[str, Any]], limit: int = 12) -> List[Dict[str, Any]]:
    if not isinstance(evidence, dict):
        return []
    feats = evidence.get("top_features")
    if not isinstance(feats, list):
        return []
    out: List[Dict[str, Any]] = []
    for f in feats:
        if not isinstance(f, dict):
            continue
        n = str(f.get("name") or "")
        raw = str(f.get("raw_name") or "")
        if _banned_feature_name(raw) or _banned_feature_name(n):
            continue
        out.append(
            {
                "name": f.get("name"),
                "raw_name": f.get("raw_name") or f.get("name"),
                "value": f.get("value"),
                "direction": f.get("direction"),
                "importance": f.get("importance"),
                "desc": f.get("desc") or f.get("interpretation"),
            }
        )
        if len(out) >= limit:
            break
    return out


def _similar_rows(similar_cases: Optional[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    if not isinstance(similar_cases, dict):
        return []
    rows = similar_cases.get("rows")
    if not isinstance(rows, list):
        return []
    out: List[Dict[str, Any]] = []
    for r in rows[:limit]:
        if not isinstance(r, dict):
            continue
        out.append(
            {
                "bill_nm": r.get("bill_nm") or r.get("bill_name"),
                "y_raw": r.get("y_raw"),
                "similarity": r.get("similarity") or r.get("score"),
                "key_deltas": r.get("key_deltas") or r.get("delta"),
            }
        )
    return out


def _summarize_similar_distribution(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    from collections import Counter

    labs = [str(r.get("y_raw") or "").strip() for r in rows if isinstance(r, dict)]
    labs = [x for x in labs if x]
    c = Counter(labs)
    top3 = c.most_common(3)
    return {"n": len(labs), "top3": [{"label": a, "count": b} for a, b in top3]}


def _call_parse(
    client: Any,
    *,
    model: str,
    system: str,
    user: str,
    schema: Any,
    temperature: float,
    max_output_tokens: int,
) -> Any:
    """responses.parse 우선, 실패 시 chat.completions + JSON 파싱 폴백."""
    try:
        resp = client.responses.parse(
            model=model,
            input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            text_format=schema,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        return resp.output_parsed
    except Exception:
        pass

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        response_format={"type": "json_object"},
        max_tokens=max_output_tokens,
    )
    txt = resp.choices[0].message.content
    data = json.loads(txt)
    return schema.model_validate(data)


# -------------------------
# minimal web search (optional)
# -------------------------

def _http_json(url: str, *, method: str = "GET", headers: Optional[Dict[str, str]] = None, data: Any = None, timeout: int = 15) -> Any:
    hdrs = headers or {}
    body = None
    if data is not None:
        if isinstance(data, (bytes, bytearray)):
            body = data
        else:
            body = json.dumps(data).encode("utf-8")
            hdrs.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=body, headers=hdrs, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        txt = resp.read().decode("utf-8", errors="ignore")
        return json.loads(txt)


def _parse_bill_date(bill: Dict[str, Any]) -> Optional[datetime]:
    for k in ("ppsl_dt", "propose_dt", "ppslDt", "proposeDt"):
        v = bill.get(k)
        if not v:
            continue
        s = str(v).strip()
        if not s:
            continue
        try:
            return datetime.fromisoformat(s.replace("Z", ""))
        except Exception:
            pass
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d")
        except Exception:
            continue
    return None


def _domain_trust(domain: str) -> float:
    d = (domain or "").lower()
    if not d:
        return 0.4
    if d.endswith(".go.kr") or d.endswith(".ac.kr") or d.endswith(".or.kr"):
        return 1.0
    majors = (
        "yonhapnews.co.kr",
        "chosun.com",
        "joongang.co.kr",
        "donga.com",
        "hani.co.kr",
        "khan.co.kr",
        "mk.co.kr",
        "news1.kr",
        "kbs.co.kr",
        "mbc.co.kr",
        "sbs.co.kr",
        "ytn.co.kr",
        "jtbc.co.kr",
        "news.naver.com",
    )
    if any(m in d for m in majors):
        return 0.9
    return 0.6


# -------------------------
# web search via OpenAI (replace Tavily/SerpAPI)
# -------------------------

def _web_search_snippets_openai(
    query: str,
    *,
    max_results: int = 8,
    allowed_domains: Optional[List[str]] = None,
    model: str = "gpt-5",
) -> List[Dict[str, str]]:
    """
    OpenAI Responses API의 web_search 툴로 웹검색 결과를 snippets 형태로 정리해서 반환.
    - 결과는 [{"title","url","snippet","source"}] 형태로 맞춰서 기존 코드와 호환.
    """
    if OpenAI is None or not _has_openai_key():
        return []

    q = (query or "").strip()
    if not q:
        return []

    client = OpenAI()

    if allowed_domains:
        tools = [{
            "type": "web_search",
            "filters": {"allowed_domains": allowed_domains[:100]},
        }]
    else:
        tools = [{"type": "web_search"}]

    system = (
        "너는 웹검색 결과를 정리하는 도우미다.\n"
        "반드시 아래 JSON 스키마만 출력해라.\n"
        "{\n"
        '  "results": [\n'
        '    {"title": "...", "url": "...", "snippet": "...", "source": "..."}\n'
        "  ]\n"
        "}\n"
        "- results는 최대 N개\n"
        "- title/url/snippet/source는 웹검색으로 확인한 내용만\n"
        "- 임의로 만들지 말 것\n"
    )
    user = (
        f"다음 쿼리로 웹검색을 수행하고, 상위 {max_results}개를 JSON으로 정리해줘.\n"
        f"query: {q}\n"
        f"N={max_results}\n"
    )

    resp = client.responses.create(
        model=model,
        tools=tools,
        tool_choice="auto",
        include=["web_search_call.action.sources"],
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    txt = resp.output_text or ""
    try:
        data = json.loads(txt)
        rows = data.get("results") or []
        out: List[Dict[str, str]] = []
        for it in rows[:max_results]:
            if not isinstance(it, dict):
                continue
            title = str(it.get("title") or "").strip()
            url = str(it.get("url") or "").strip()
            snippet = str(it.get("snippet") or "").strip()
            source = str(it.get("source") or "").strip()
            if not (title or snippet or url):
                continue
            out.append({
                "title": title[:200],
                "url": url[:500],
                "snippet": snippet[:400],
                "source": source[:120] or (urllib.parse.urlparse(url).netloc if url else "unknown"),
            })
        return out
    except Exception:
        return []


def _heuristic_social_score(results: List[Dict[str, str]]) -> float:
    """
    키가 없거나 LLM이 실패했을 때를 위한 매우 보수적 휴리스틱(0~100).
    - 출처 수 + 출처 신뢰도(도메인) + 논쟁/대중성 키워드 기반
    """
    if not results:
        return 10.0

    n = len(results)
    trusts = []
    text_blob = ""
    domains = set()
    for r in results:
        dom = str(r.get("source") or "")
        domains.add(dom)
        trusts.append(_domain_trust(dom))
        text_blob += " " + str(r.get("title") or "") + " " + str(r.get("snippet") or "")

    avg_trust = sum(trusts) / max(1, len(trusts))
    diversity = min(1.0, len(domains) / 5.0)
    volume = min(1.0, n / 8.0)

    low = text_blob.lower()
    kw = ["논란", "파장", "찬반", "반발", "청원", "시위", "사회", "국민", "전국", "쟁점", "갈등"]
    hit = sum(1 for k in kw if k in low)
    kw_score = min(1.0, hit / 6.0)

    raw01 = 0.15 + 0.45 * volume + 0.20 * avg_trust + 0.10 * diversity + 0.10 * kw_score
    raw01 = max(0.0, min(1.0, raw01))
    return float(round(raw01 * 100.0, 2))


# -------------------------
# payload builders
# -------------------------

def build_min_payload(
    *,
    bill: Dict[str, Any],
    pred: Dict[str, Any],
    evidence: Optional[Dict[str, Any]] = None,
    similar_cases: Optional[Dict[str, Any]] = None,
    model_info: Optional[Dict[str, Any]] = None,
    summary_lines: Optional[List[str]] = None,
) -> Dict[str, Any]:
    bill_min = {
        "bill_no": bill.get("bill_no") or bill.get("billNo"),
        "bill_name": bill.get("bill_name") or bill.get("bill_nm") or bill.get("bill_name"),
        "committee_name": bill.get("committee_name") or bill.get("curr_committee_name"),
        "proposer_name": bill.get("proposer_name") or bill.get("ppsr_nm"),
        "proposer_kind": bill.get("proposer_kind"),
        "proposer_count_est": bill.get("proposer_count_est"),
        "amendment_type": bill.get("amendment_type"),
        "budget_yn": bill.get("budget_yn") if "budget_yn" in bill else bill.get("budget"),
        "elapsed_days": bill.get("elapsed_days"),
        "summary": _clip_text(bill.get("summary"), 1600),
        "summary_lines": summary_lines or [],
    }

    pred_min = {
        "final_pred_8": pred.get("final_pred_8") or pred.get("final_pred"),
        "confidence": (pred.get("insights") or {}).get("confidence"),
        "p_fail_step1": pred.get("p_fail_step1"),
        "p_non_alt": pred.get("p_non_alt"),
        "p_orig": pred.get("p_orig"),
        "p_alt": pred.get("p_alt"),
        "thresholds": pred.get("thresholds") or {},
        "model_version": pred.get("model_version"),
    }

    top_features = _select_top_features(evidence, limit=12)
    rows_min = _similar_rows(similar_cases, limit=10)

    return {
        "bill": bill_min,
        "prediction": pred_min,
        "evidence_hint": {"top_features": top_features},
        "similar_cases": {
            "method": (similar_cases or {}).get("method", "-") if isinstance(similar_cases, dict) else "-",
            "filters": (similar_cases or {}).get("filters", []) if isinstance(similar_cases, dict) else [],
            "top_k": (similar_cases or {}).get("top_k", len(rows_min)) if isinstance(similar_cases, dict) else len(rows_min),
            "rows": rows_min,
            "distribution": _summarize_similar_distribution(rows_min),
        },
        "model_info": model_info or {},
    }


# -------------------------
# section prompts
# -------------------------

def _sys_common() -> str:
    return (
        "너는 '법안 처리결과 예측 보고서'의 문장화/요약 보조다.\n"
        "반드시 아래 원칙을 지켜라:\n"
        "- 입력 payload의 사실 데이터만 사용(추정/창작 금지)\n"
        "- 과장/인과 단정 금지: '가능한 해석/시사점' 정도로만 표현\n"
        "- 모델/확률/점수/임계값/시스템 내부를 설명하지 말 것(사용자 관점 문장만)\n"
        "- 출력은 반드시 JSON(스키마에 맞춤), 한국어, 공손체(~습니다/~합니다)로 통일\n"
    )


def _prompt_sec1(payload: Dict[str, Any]) -> Tuple[str, str]:
    system = _sys_common() + (
        "너의 목표는 1번 섹션의 '요약'을 만드는 것이다.\n"
        "- bill.summary만 근거로 사용\n"
        "- 정확히 5개 문장\n"
        "- 각 문장은 1줄로, 핵심 조치/변경점을 사용자에게 설명하는 형태\n"
        "- 변수명/수치 나열은 피하고, 꼭 필요한 경우만 간단히 포함\n"
    )
    user = (
        "아래 payload를 바탕으로 summary_lines 5개를 만들어줘.\n"
        "payload:\n" + _safe_json({"bill": payload.get("bill")})
    )
    return system, user


def _prompt_sec0(payload: Dict[str, Any]) -> Tuple[str, str]:
    system = _sys_common() + (
        "너의 목표는 0번 섹션의 '핵심 근거(Top 3)'를 사용자 관점에서 한 줄씩 만드는 것이다.\n"
        "- 근거는 evidence_hint.top_features에서만 뽑아라.\n"
        "- '긍정적/부정적 영향'처럼 인과를 단정하는 표현을 피하고, '시사합니다/참고 신호입니다'처럼 완곡하게 표현한다.\n"
        "- 숫자/변수명 그대로 나열 금지(의미를 말로 풀기)\n"
        "- 정확히 3개, 각 1문장(너무 길지 않게)\n"
    )
    user = (
        "다음 payload를 바탕으로 top_reasons 3개를 생성해줘.\n"
        "payload:\n" + _safe_json(
            {"bill": payload.get("bill"), "prediction": payload.get("prediction"), "evidence_hint": payload.get("evidence_hint")}
        )
    )
    return system, user


def _prompt_sec3(payload: Dict[str, Any]) -> Tuple[str, str]:
    system = _sys_common() + (
        " 역할: 너는 대한민국 국회 입법 과정과 통계 데이터를 분석하는 '수석 입법 연구원'이다.\n"
        "\n"
        "## 목표: 3번 섹션 '해석 요약(process_summary)' 작성\n"
        "사용자가 제공한 피처(Feature) 데이터를 바탕으로, 해당 법안이 모델 내에서 어떻게 해석되었는지 입법 맥락을 담아 요약하라.\n"
        "\n"
        "## 작성 규칙 (반드시 지킬 것):\n"
        "1. 분석적 톤 유지: '참고됩니다', '도움이 됩니다' 같은 당연한 말 대신 '시그널', '경향성', '우선순위', '영향력' 등 전문 용어를 써라.\n"
        "2. 구조(2~4문장):\n"
        "   - [1문장]: 상위 2~3개 피처(한글명)를 언급하며, 이것들이 입법 데이터상 어떤 통계적 유의미함을 갖는지 요약.\n"
        "   - [2~3문장]: 구체적인 '값'과 '해석(desc)'을 연결해, 입법 절차적 관점에서 설명.\n"
        "   - [마지막]: 결측치('-')가 있다면 '데이터 공백으로 인한 보수적 해석 필요' 언급.\n"
        "3. 금지사항: PASS/FAIL, 점수/확률 숫자, 최종 라벨명, 영문 피처명 직접 노출 금지.\n"
        "\n"
        "## 입법 맥락 가이드 (설명 방식 참고):\n"
        "- 위원장 발의/대안: '개별 의원 발의보다 정당 간 합의 수준이 높고 처리 가능성이 큰 상태'로 해석.\n"
        "- 경과일/기간: '회기 내 처리의 시급성이나 논의의 성숙도'로 해석.\n"
        "- 소관위원회: '해당 위원회의 과거 법안 처리 성향이 반영됨'으로 해석.\n"
        "\n"
        "## 좋은 예시와 나쁜 예시:\n"
        "- (나쁜 예): 발의 주체가 위원장인 점은 참고됩니다. 법안명이 무엇인지도 중요합니다.\n"
        "- (좋은 예): 발의 주체와 대안 여부는 과거 입법 패턴상 처리 우선순위를 결정짓는 핵심 시그널입니다. 특히 위원장 제안 법안은 소관 위원회 내부의 합의가 상당 부분 완료되었음을 시사하며, 이는 최종 의결 단계에서의 긍정적 경향성으로 연결됩니다.\n"
    )

    bill = payload.get("bill") or {}
    bill_meta = {
        "bill_id": bill.get("bill_id") or bill.get("billId"),
        "bill_no": bill.get("bill_no") or bill.get("billNo"),
        "bill_nm": bill.get("bill_nm") or bill.get("bill_name"),
        "committee_name": bill.get("committee_name") or bill.get("curr_committee_name"),
        "proposer_name": bill.get("ppsr_nm") or bill.get("proposer_name"),
        "proposer_kind": bill.get("proposer_kind"),
        "amendment_type": bill.get("amendment_type"),
        "budget_yn": bill.get("budget_yn") if "budget_yn" in bill else bill.get("budget"),
        "elapsed_days": bill.get("elapsed_days"),
    }

    evidence_hint = payload.get("evidence_hint") or {}
    top_features = evidence_hint.get("top_features")
    if isinstance(top_features, list):
        top_features = top_features[:8]

    user = (
        "payload를 참고해 process_summary(2~4문장)를 작성해줘.\n"
        "아래 정보만 근거로 사용해. 특히 top_features의 desc/interpretation을 기반으로 써.\n"
        "payload:\n"
        + _safe_json(
            {
                "bill_meta": bill_meta,
                "top_features": top_features,
            }
        )
    )
    return system, user


def _prompt_sec3_table(rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    system = _sys_common() + (
        "다음 rows를 보고, 각 행(rank|feature|value)에 대해 '해석' 1문장씩 생성해라.\n"
        "- 문장은 공손체(~습니다/~합니다)\n"
        "- 확률/점수/임계값/최종라벨 언급 금지\n"
        "- 영문 raw 변수명은 노출하지 말 것\n"
        "- 결과는 row_interpretations 리스트로, rows와 동일한 순서/개수로 출력\n"
        "- value가 '-'이면 '근거가 제한되어 보수적 해석이 필요'처럼 1문장으로 처리\n"
        "-(중요) 이 feature가 rank만큼 중요하면 어떤의미인지, feature가 value 값인게 의안처리결과에 어떠한 영향이있을지 중점으로 사례기반으로 작성\n"
        "- value 가 1인경우 긍정의의미(있다,했다)의 의미임.(예: feature가 상임 위원회 소속 여부 이고 value가 1 이면 의원이 해당 상임위원회에 소속되었다라는 뜻)\n"
    )
    user = "다음 rows에 대해 각 행 해석 1문장씩 생성해줘.\nrows:\n" + _safe_json({"rows": rows})
    return system, user


def _prompt_sec4(payload: Dict[str, Any]) -> Tuple[str, str]:
    system = _sys_common() + (
        "너의 목표는 4번 섹션의 '유사사례 결과·차이 기반 해석'을 위한 힌트를 만드는 것이다.\n"
        "- similar_cases.distribution, similar_cases.rows[*].y_raw, key_deltas만 근거로 사용\n"
        "- common_patterns: 유사 법안들의 실제 결과/맥락에서 자주 보인 패턴 2~3개\n"
        "- diff_patterns: 이번 법안의 핵심 차이로 본 점검 포인트 2~3개\n"
        "- similar_conclusion: 종합 1~2문장(단정 금지)\n"
        "- 변수명/코드명 직접 노출 최소화\n"
    )
    user = (
        "아래 payload 기반으로 common_patterns, diff_patterns, similar_conclusion을 작성해줘.\n"
        "payload:\n" + _safe_json({"similar_cases": payload.get("similar_cases")})
    )
    return system, user


def _prompt_sec4_row_deltas(rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    system = _sys_common() + (
        "너의 목표는 4번 섹션 '유사 사례 비교분석' 표에서, 각 행의 '핵심 차이'를 서술형으로 바꾸는 것이다.\n"
        "\n"
        "작성 규칙(엄수):\n"
        "- 입력 rows[*].deltas_human만 근거로 사용(추정 금지)\n"
        "- 각 행당 정확히 1문장, 공손체(~습니다/~합니다)\n"
        "- 분석 톤: '차이입니다/달라집니다/변화가 관측됩니다/시사합니다' 같은 표현 활용\n"
        "- 숫자/확률/유사도 언급 금지 (발의자 수 차이는 예외)\n"
        "- 결과 라벨(y_raw) 직접 언급 금지(표에 이미 있음)\n"
        "- deltas_human가 '-'이면 출력도 '눈에 띄는 차이가 없습니다.'로\n"
        "- 같은 의미 반복 금지(핵심만)\n"
        "\n"
        "좋은 예:\n"
        "- \"발의자 수가 8명 더 많고 발의 주체가 위원장 대신 의원 발의로 바뀐 점, 당시 정당이 달라진 점이 핵심 차이입니다.\"\n"
    )
    user = "다음 rows에 대해 row_delta_texts를 생성해줘.\nrows:\n" + _safe_json({"rows": rows})
    return system, user


# ✅ NEW: 사회적 영향력 분석 프롬프트
def _prompt_sec5_social(
    *,
    bill_min: Dict[str, Any],
    propose_dt: Optional[str],
    search_results: List[Dict[str, str]],
) -> Tuple[str, str]:
    system = _sys_common() + (
        "너의 목표는 '5. 사회적 영향력 분석'을 작성하는 것이다.\n"
        "\n"
        "출력 형식:\n"
        "- social_impact_score: 0~100 연속 점수(정수/실수 모두 가능)\n"
        "- social_impact_lines: 정확히 3개 문장(각 1줄, 공손체)\n"
        "- sources: 2~5개(문자열). 반드시 입력 search_results의 source/title에서만 고른다.\n"
        "\n"
        "점수 산정(발의일 당시 기준) 가이드:\n"
        "- 출처 수: 1곳 < 2~3곳 < 다수\n"
        "- 출처 신뢰도: 공공기관·주요 언론·학술 자료 > 기타\n"
        "- 대중성: 전국적 관심 > 광범위한 사회집단 > 제한적 집단\n"
        "- 논쟁성: 찬반 갈등, 정책적 쟁점, 사회적 논의의 존재 여부\n"
        "\n"
        "주의:\n"
        "- search_results에 없는 사실/출처/사건을 만들어내지 말 것.\n"
        "- search_results가 빈 경우: 점수는 낮게(예: 0~25) 주고 '자료 부족'을 명시할 것.\n"
        "- social_issue_yn과는 독립적으로 판단한다.\n"
        "- 모델/확률/임계값/최종라벨 언급 금지.\n"
    )

    user = (
        "다음 정보를 바탕으로 사회적 영향력 분석을 작성해줘.\n"
        f"- 발의일(가능하면 참고): {propose_dt or '-'}\n"
        "bill:\n"
        + _safe_json({"bill": bill_min})
        + "\nsearch_results(발의일 전후 기사/자료 스니펫):\n"
        + _safe_json({"search_results": search_results})
    )
    return system, user


def _prompt_sec5(payload: Dict[str, Any]) -> Tuple[str, str]:
    system = _sys_common() + (
        "너의 목표는 6번 섹션(시사점: 영향도)을 입법·정책 분석 관점에서 작성하는 것이다.\n"
        "- 핵심 구조는 반드시 '주요 변화(As-is→To-be) → 영향(대상/파급) → 문제 해결 가능성(해결될지/부작용)' 순서로 쓴다.\n"
        "- 단정 금지: '~일 수 있습니다/가능성이 있습니다/검토해볼 필요가 있습니다'를 사용한다.\n"
        "- 숫자/금액/통계는 입력에 있는 것만 사용(없으면 절대 생성 금지).\n"
        "- 모델 버전/임계값/피처명/SHAP 등 기술 설명 금지.\n"
        "- 출력은 next_actions_md 3개이며, 각 항목은 아래 형식을 반드시 지킨다:\n"
        "  1) 루트 bullet 1줄: '- **[대상/영향 키워드]**: ...'\n"
        "  2) 하위 bullet 3줄:\n"
        "     '  - 무엇이 어떻게 바뀌는지(As-is→To-be)'\n"
        "     '  - 누구에게 어떤 변화가 생기는지(영향)'\n"
        "     '  - 해결될지/부작용이 생길지(문제 해결)'\n"
        "- 각 항목은 1개의 주제만 다룬다(중복 금지).\n"
        "- 하위 bullet에 (As-is→To-be),(영향),(문제 해결) 같은 라벨 텍스트는 출력하지 말 것(문장으로 풀어쓸 것).\n"
    )

    user = (
        "아래 payload를 바탕으로 섹션6 시사점(영향도) next_actions_md 3개를 작성해줘.\n"
        "가능하면 bill.summary_lines를 우선 사용하고, 없으면 bill.summary에서 핵심 변화들을 뽑아 써.\n"
        "payload:\n" + _safe_json({"bill": payload.get("bill")})
    )
    return system, user



def _delta_key_ko(key: str) -> str:
    k = (key or "").strip()
    if not k:
        return k
    return _DELTA_KEY_KO.get(k, k)


def _drop_elapsed_related_token(s: str) -> bool:
    low = (s or "").strip().lower()
    return ("elapsed" in low) or ("경과" in low)


def _humanize_deltas_for_llm(deltas: Any) -> str:
    if deltas is None:
        return "-"
    if isinstance(deltas, list):
        parts: List[str] = []
        for x in deltas:
            ss = str(x).strip()
            if not ss or ss == "-":
                continue
            if _drop_elapsed_related_token(ss):
                continue
            parts.append(ss)
        return "; ".join(parts) if parts else "-"

    raw = str(deltas).strip()
    if not raw or raw == "-":
        return "-"

    chunks = re.split(r"\s*;\s*|\s*,\s*|\s*\|\s*", raw)
    out_parts: List[str] = []
    for c in chunks:
        s = c.strip()
        if not s or s == "-":
            continue
        if _drop_elapsed_related_token(s):
            continue

        m = _RE_DELTA_PLUS.match(s)
        if m:
            key, sign, num = m.group(1), m.group(2), m.group(3)
            out_parts.append(f"{_delta_key_ko(key)} {sign}{num}")
            continue

        m = _RE_DELTA_ARROW.match(s)
        if m:
            key, a, b = m.group(1), m.group(2).strip(), m.group(3).strip()
            out_parts.append(f"{_delta_key_ko(key)}: {a}→{b}")
            continue

        m = _RE_DELTA_KV.match(s)
        if m:
            key, rest = m.group(1), m.group(2).strip()
            if _drop_elapsed_related_token(rest):
                continue
            out_parts.append(f"{_delta_key_ko(key)}: {rest}")
            continue

        out_parts.append(s)

    return "; ".join(out_parts) if out_parts else "-"


# -------------------------
# main API
# -------------------------

def enrich_for_report(
    *,
    bill: Dict[str, Any],
    pred: Dict[str, Any],
    evidence: Optional[Dict[str, Any]] = None,
    similar_cases: Optional[Dict[str, Any]] = None,
    timeline: Any = None,
    model_info: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """섹션별 LLM enrich 수행."""
    evidence_out = dict(evidence or {})
    # ✅ 중요: report_builder가 pred.insights를 먼저 합치므로,
    # LLM이 실패해도 pred.insights.next_actions(예: log_elapsed 안내)가 그대로 노출되지 않게
    # 기본값(빈 리스트)로 덮어쓸 준비를 해둡니다.
    evidence_out.setdefault("next_actions", [])


    # --- propose date & query build (웹검색용) ---
    propose_dt = _parse_bill_date(bill)
    propose_dt_str = propose_dt.strftime("%Y-%m-%d") if propose_dt else None

    bill_nm = str(bill.get("bill_nm") or bill.get("bill_name") or "").strip()
    bill_no = str(bill.get("bill_no") or bill.get("billNo") or "").strip()
    yymm = f"{propose_dt.year}년 {propose_dt.month}월" if propose_dt else ""

    q_base = bill_nm or bill_no or "법률안"
    q1 = f"{q_base} {yymm} 발의 논란"
    q2 = f"{q_base} {yymm} 기사"

    # ✅ OpenAI web_search: 키가 있을 때만 시도
    search_results: List[Dict[str, str]] = []
    if OpenAI is not None and _has_openai_key():
        try:
            r1 = _web_search_snippets_openai(q1, max_results=6, model="gpt-5")
            r2 = _web_search_snippets_openai(q2, max_results=6, model="gpt-5")

            seen = set()
            merged: List[Dict[str, str]] = []
            for r in (r1 + r2):
                u = str(r.get("url") or "")
                if u and u in seen:
                    continue
                if u:
                    seen.add(u)
                merged.append(r)
            search_results = merged[:8]
        except Exception:
            search_results = []
    else:
        search_results = []

    # 키가 없거나 OpenAI 미사용이면: 휴리스틱 점수 + 보수적 문장 3줄만 채움
    if OpenAI is None or not _has_openai_key():
        score = _heuristic_social_score(search_results)
        evidence_out["social_impact_score"] = score
        evidence_out["social_impact_lines"] = [
            "발의일 전후로 확인 가능한 공개 자료의 양과 출처 신뢰도를 기준으로 사회적 이슈화 강도를 보수적으로 산정했습니다.",
            "다만 검색 가능한 출처가 제한적일 수 있어, 실제 사회적 파장과의 차이가 발생할 수 있습니다.",
            "추가적인 주요 언론 보도·공공기관 자료·공식 논의 기록이 확인되면 점수 보정이 가능합니다.",
        ]
        srcs = []
        for r in search_results:
            s = str(r.get("source") or "").strip()
            t = str(r.get("title") or "").strip()
            if s and s not in srcs:
                srcs.append(s)
            elif t and t not in srcs:
                srcs.append(t[:60])
            if len(srcs) >= 5:
                break
        evidence_out["social_impact_sources"] = srcs

        return {
            "evidence": evidence_out,
            "similar_cases": dict(similar_cases or {}) if isinstance(similar_cases, dict) else similar_cases,
        }

    client = OpenAI()
    model_name = model or os.getenv("OPENAI_REPORT_MODEL", "gpt-4.1")

    payload = build_min_payload(
        bill=bill,
        pred=pred,
        evidence=evidence,
        similar_cases=similar_cases,
        model_info=model_info,
        summary_lines=None,
    )

    summary_lines: Optional[List[str]] = None
    try:
        s, u = _prompt_sec1(payload)
        sec1 = _call_parse(client, model=model_name, system=s, user=u, schema=Sec1Out, temperature=0.2, max_output_tokens=260)
        summary_lines = [str(x).strip() for x in (sec1.summary_lines or []) if str(x).strip()]
        summary_lines = (summary_lines or [])[:5]
        if len(summary_lines) == 5:
            evidence_out["bill_summary_lines"] = summary_lines
    except Exception:
        summary_lines = None

    if summary_lines:
        payload = build_min_payload(
            bill=bill,
            pred=pred,
            evidence=evidence_out,
            similar_cases=similar_cases,
            model_info=model_info,
            summary_lines=summary_lines,
        )

    try:
        s, u = _prompt_sec0(payload)
        sec0 = _call_parse(client, model=model_name, system=s, user=u, schema=Sec0Out, temperature=0.2, max_output_tokens=260)
        evidence_out["top_reasons"] = [str(x).strip() for x in (sec0.top_reasons or []) if str(x).strip()][:3]
    except Exception:
        pass

    try:
        s, u = _prompt_sec3(payload)
        sec3 = _call_parse(client, model=model_name, system=s, user=u, schema=Sec3Out, temperature=0.2, max_output_tokens=340)
        evidence_out["process_summary"] = str(sec3.process_summary).strip()
    except Exception:
        pass

    # ✅ 섹션 3 표의 '해석'을 LLM으로 한 줄 생성해서 top_features.desc에 주입
    try:
        feats_for_table = _select_top_features(evidence_out, limit=12)
        if feats_for_table:
            rows: List[Dict[str, Any]] = []
            for i, f in enumerate(feats_for_table, start=1):
                name = str(f.get("name") or f.get("raw_name") or "-").strip() or "-"
                raw_name = str(f.get("raw_name") or "").strip().lower()
                v = f.get("value")

                if name == "법안 개요" or raw_name == "summary":
                    value_disp = "제안 이유 및 주요 내용" if (v is not None and str(v).strip()) else "-"
                else:
                    if v is None:
                        value_disp = "-"
                    else:
                        sv = str(v).strip()
                        value_disp = "-" if sv.lower() in MISSING_STRS else (sv if len(sv) <= 80 else sv[:79] + "…")

                rows.append({"rank": i, "feature": name, "value": value_disp})

            s, u = _prompt_sec3_table(rows)
            sec3t = _call_parse(client, model=model_name, system=s, user=u, schema=Sec3TableOut, temperature=0.2, max_output_tokens=520)
            interps = [str(x).strip() for x in (sec3t.row_interpretations or []) if str(x).strip()]

            if len(interps) == len(feats_for_table):
                mapping: Dict[str, str] = {}
                for f, it in zip(feats_for_table, interps):
                    key = str(f.get("raw_name") or f.get("name") or "").strip()
                    if key:
                        mapping[key] = it

                feats_all = evidence_out.get("top_features")
                if isinstance(feats_all, list):
                    for fx in feats_all:
                        if not isinstance(fx, dict):
                            continue
                        key = str(fx.get("raw_name") or fx.get("name") or "").strip()
                        if key in mapping:
                            fx["desc"] = mapping[key]
                            fx["interpretation"] = mapping[key]
    except Exception:
        pass

    # ✅ NEW: 섹션 5 사회적 영향력 분석(웹검색 스니펫 기반)
    try:
        bill_min = payload.get("bill") or {}
        s, u = _prompt_sec5_social(bill_min=bill_min, propose_dt=propose_dt_str, search_results=search_results)
        sec5s = _call_parse(client, model=model_name, system=s, user=u, schema=Sec5SocialOut, temperature=0.2, max_output_tokens=420)

        score = float(sec5s.social_impact_score)
        if score != score:  # NaN
            raise ValueError()

        score = max(0.0, min(100.0, score))
        lines = [str(x).strip() for x in (sec5s.social_impact_lines or []) if str(x).strip()]
        lines = (lines + ["", "", ""])[:3]
        sources = [str(x).strip() for x in (sec5s.sources or []) if str(x).strip()]
        sources = sources[:5]

        evidence_out["social_impact_score"] = score
        evidence_out["social_impact_lines"] = lines[:3]
        evidence_out["social_impact_sources"] = sources
    except Exception:
        score = _heuristic_social_score(search_results)
        evidence_out["social_impact_score"] = score
        evidence_out["social_impact_lines"] = [
            "발의일 전후로 확인 가능한 공개 자료의 양과 출처 신뢰도를 기준으로 사회적 이슈화 강도를 보수적으로 산정했습니다.",
            "다만 검색 가능한 출처가 제한적일 수 있어, 실제 사회적 파장과의 차이가 발생할 수 있습니다.",
            "추가적인 주요 언론 보도·공공기관 자료·공식 논의 기록이 확인되면 점수 보정이 가능합니다.",
        ]
        srcs = []
        for r in search_results:
            s = str(r.get("source") or "").strip()
            if s and s not in srcs:
                srcs.append(s)
            if len(srcs) >= 5:
                break
        evidence_out["social_impact_sources"] = srcs

    # ✅ (기존) 향후 대응 방향(next_actions) 생성
    try:
        s, u = _prompt_sec5(payload)
        sec5 = _call_parse(
            client,
            model=model_name,
            system=s,
            user=u,
            schema=Sec5Out,
            temperature=0.2,
            max_output_tokens=650,
        )
        md_items = [str(x).strip() for x in (sec5.next_actions_md or []) if str(x).strip()]
        evidence_out["next_actions"] = md_items[:3]
    except Exception:
        # ✅ 실패해도 pred.insights의 옛 next_actions가 보이지 않도록 빈 리스트 유지
        evidence_out["next_actions"] = []

    if isinstance(similar_cases, dict):
        similar_cases_out = dict(similar_cases)
    else:
        similar_cases_out = {"method": "-", "filters": [], "top_k": 0, "rows": similar_cases or []}

    # ✅ (요청 반영) 섹션 4 표의 '핵심 차이'를 서술형으로 변환하여 rows[*].key_deltas_text에 주입
    try:
        rows_src = similar_cases_out.get("rows")
        if isinstance(rows_src, list) and rows_src:
            rows_for_prompt: List[Dict[str, Any]] = []
            for r in rows_src[:10]:
                if not isinstance(r, dict):
                    continue
                kd = r.get("key_deltas") or r.get("delta")
                kd_h = _humanize_deltas_for_llm(kd)
                rows_for_prompt.append(
                    {
                        "bill_nm": r.get("bill_nm") or r.get("bill_name"),
                        "deltas_human": kd_h,
                    }
                )

            if rows_for_prompt:
                s, u = _prompt_sec4_row_deltas(rows_for_prompt)
                sec4r = _call_parse(client, model=model_name, system=s, user=u, schema=Sec4RowDeltaOut, temperature=0.2, max_output_tokens=520)
                texts = [str(x).strip() for x in (sec4r.row_delta_texts or [])]

                if len(texts) == len(rows_for_prompt):
                    j = 0
                    for r in rows_src[:10]:
                        if not isinstance(r, dict):
                            continue
                        t = texts[j].strip() if j < len(texts) else "-"
                        if rows_for_prompt[j].get("deltas_human") == "-":
                            t = "-"
                        r["key_deltas_text"] = t if t else "-"
                        j += 1
    except Exception:
        pass

    return {
        "evidence": evidence_out,
        "similar_cases": similar_cases_out,
    }
