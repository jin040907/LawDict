from __future__ import annotations

"""backend/report_insights.py

보고서용 근거/해설(휴리스틱 + SHAP) 생성기.

왜 이 파일이 필요한가
- report_builder는 0번 섹션(Top3)과 3번 섹션(피처 표)을 `evidence.top_reasons/top_features`에서 읽는다.
- sample_input이나 inference payload에 evidence/insights가 없으면 보고서에
  "(근거 데이터가 부족...)" / "(피처 기반 근거 데이터가 없습니다...)"가 출력된다.

설계 원칙
- 확률(p_*)/임계값(thr_*)/라벨(final_pred_*)은 근거 피처로 쓰지 않는다.
- 가능하면 CatBoost SHAP(인스턴스별 기여도)로, 불가하면 global importance로 폴백.
- SHAP 계산이 불가하거나 모델을 못 찾는 경우에도 '최소한의 의미있는 메타 피처'로 폴백하여
  Top3/TopFeatures가 비지 않게 한다.

주의
- SHAP/importance는 설명을 위한 근사치이며, 인과를 단정하지 않는다.
"""

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from catboost import Pool
except Exception:  # pragma: no cover
    Pool = None  # type: ignore


PASS_SET = {"원안가결", "수정가결", "대안반영폐기", "수정안반영폐기"}
FAIL_SET = {"부결", "폐기", "철회", "임기만료폐기"}

MISSING_STRS = {"", "none", "null", "nan", "na", "n/a", "-", "없음"}


# -------------------------
# basic utils
# -------------------------

def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and (math.isnan(v) or np.isnan(v)):
        return True
    if isinstance(v, str):
        return v.strip().lower() in MISSING_STRS
    return False


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if v != v:
            return None
        return v
    except Exception:
        return None


def _format_value(v: Any) -> Any:
    """보고서 표시용 value 정리: 1.0 -> 1"""
    try:
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating, float)):
            fv = float(v)
            if fv.is_integer():
                return int(fv)
            return fv
    except Exception:
        pass
    return v


def _is_banned_feature(name: str) -> bool:
    """확률/임계값/라벨 등 '모델 중간산출물'이 근거 피처로 노출되는 것을 방지."""
    low = (name or "").strip().lower()
    if not low:
        return True
    banned_prefix = (
        "p_",
        "thr_",
        "threshold",
        "final_pred",
        "model_",
        "fail_topk",
    )
    banned_exact = {"final_pred_8", "p_fail_step1", "p_non_alt", "p_orig", "p_alt"}
    if low in banned_exact:
        return True
    if any(low.startswith(p) for p in banned_prefix):
        return True
    return False


def _human_feature_name(raw: str) -> str:
    mapping = {
        # 시간/절차
        "elapsed_days": "경과일",
        "committee_referral_days": "위원회 회부까지 기간",
        "committee_meeting_days": "위원회 심사기간",
        # 발의/정치
        "proposer_count_est": "발의자수",
        "proposer_kind": "발의 주체",
        "ppsr_kind": "발의 주체",
        "party_at_term": "당(대수 기준)",
        "n_negotiation_groups": "협상그룹 수",
        # 성격
        "amendment_type": "개정유형",
        "alternative_yn": "대안 여부",
        "is_alternative": "대안 여부",
        "budget_yn": "예산/비용추계 여부",
        "budget": "예산/비용추계 여부",
        # 텍스트
        "summary_embed": "요약 텍스트 패턴(임베딩)",
        "billname_embed": "의안명 텍스트 패턴(임베딩)",
    }
    return mapping.get(raw, raw)


def _group_feature(name: str) -> Optional[str]:
    """임베딩 차원이 많을 때 Top이 임베딩으로 도배되는 걸 방지."""
    if name.startswith("billname_embed"):
        return "billname_embed"
    if name.startswith("summary_embed"):
        return "summary_embed"
    if name.endswith("_embed"):
        return "text_embed"
    return None


def _cat_features_from_df(df: pd.DataFrame) -> List[int]:
    cat_idx: List[int] = []
    for i, c in enumerate(df.columns):
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]):
            cat_idx.append(i)
    return cat_idx


def _align_X_to_model_features(X_one: pd.DataFrame, model: Any) -> pd.DataFrame:
    feat_names = list(getattr(model, "feature_names_", None) or getattr(model, "feature_names_", None) or model.feature_names_)
    missing = [c for c in feat_names if c not in X_one.columns]
    if missing:
        X_one = pd.concat([X_one, pd.DataFrame(np.nan, index=X_one.index, columns=missing)], axis=1)
    X_one = X_one[feat_names].copy()
    return X_one


def _try_instance_shap_topk(
    model: Any,
    X_one: pd.DataFrame,
    target_class_idx: Optional[int],
    k: int = 24,
) -> List[Tuple[str, float]]:
    """인스턴스 1개에 대한 SHAP top-k (가능한 경우)."""
    if Pool is None:
        return []

    X_aligned = _align_X_to_model_features(X_one, model)
    cat_idx = _cat_features_from_df(X_aligned)
    pool = Pool(X_aligned, cat_features=cat_idx) if cat_idx else Pool(X_aligned)

    shap = model.get_feature_importance(pool, type="ShapValues")
    shap = np.asarray(shap)

    feat_names = list(X_aligned.columns)

    if shap.ndim == 2:
        vals = shap[0, :-1]
    elif shap.ndim == 3:
        if target_class_idx is None:
            try:
                target_class_idx = int(np.argmax(model.predict_proba(X_aligned)[0]))
            except Exception:
                target_class_idx = 0
        vals = shap[0, target_class_idx, :-1]
    else:
        return []

    pairs = [(feat_names[i], float(vals[i])) for i in range(len(feat_names))]
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:k]


def _try_global_importance_topk(model: Any, k: int = 24) -> List[Tuple[str, float]]:
    """모델 전체 기반 importance top-k (sign 없음)."""
    try:
        imp = model.get_feature_importance(type="PredictionValuesChange")
        imp = np.asarray(imp).reshape(-1)
        feat_names = list(getattr(model, "feature_names_", None) or model.feature_names_)
        pairs = [(feat_names[i], float(imp[i])) for i in range(len(feat_names))]
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        return pairs[:k]
    except Exception:
        return []


def _pick_model(rt: Any, key: str) -> Optional[Any]:
    """runtime에서 step 모델을 찾아온다.

    환경/버전마다 저장되는 키가 달라서 (step2a vs step2A 등) 케이스를 넓게 커버한다.
    """

    # 1) dict 계열 우선
    for dname in ("models", "model_map", "clf_map"):
        d = getattr(rt, dname, None)
        if isinstance(d, dict):
            if key in d:
                return d[key]
            # 대소문자/별칭
            alt_keys = {
                "step1": ["step1", "Step1", "STEP1"],
                "step2a": ["step2a", "step2A", "Step2A", "STEP2A"],
                "step2b": ["step2b", "step2B", "Step2B", "STEP2B"],
                "step2c": ["step2c", "step2C", "Step2C", "STEP2C"],
                "stepf": ["stepf", "stepF", "StepF", "STEPF"],
            }.get(key, [])
            for k2 in alt_keys:
                if k2 in d:
                    return d[k2]

    # 2) attribute 후보
    candidates = [
        f"{key}_model",
        f"{key}_clf",
        f"cbm_{key}",
        key,
        "step1_model" if key == "step1" else None,
        "step2a_model" if key == "step2a" else None,
        "step2A_model" if key == "step2a" else None,
        "step2b_model" if key == "step2b" else None,
        "step2B_model" if key == "step2b" else None,
        "step2c_model" if key == "step2c" else None,
        "step2C_model" if key == "step2c" else None,
        "stepf_model" if key == "stepf" else None,
        "stepF_model" if key == "stepf" else None,
    ]

    for name in candidates:
        if not name:
            continue
        m = getattr(rt, name, None)
        if m is not None:
            return m

    # 3) 마지막 폴백: runtime에 single model이 있는 경우
    for name in ("model", "clf", "final_model"):
        m = getattr(rt, name, None)
        if m is not None:
            return m

    return None


def _thr_get(th: Dict[str, Any], *keys: str, default: float = 0.5) -> float:
    for k in keys:
        if k in th:
            v = _safe_float(th.get(k))
            if v is not None:
                return float(v)
    return float(default)


def _get_prob(pred: Dict[str, Any], *keys: str) -> Optional[float]:
    for k in keys:
        if k in pred:
            v = _safe_float(pred.get(k))
            if v is not None:
                return float(v)
    return None


def _decide_path(pred: Dict[str, Any], thresholds: Dict[str, Any]) -> List[Tuple[str, str, Optional[int]]]:
    """결정 경로에 해당하는 step 모델만 설명에 사용."""
    p_fail = _get_prob(pred, "p_fail_step1", "p_fail", "p_fail_step")
    p_non_alt = _get_prob(pred, "p_non_alt")
    p_orig = _get_prob(pred, "p_orig")
    p_alt = _get_prob(pred, "p_alt")

    thr_fail = _thr_get(thresholds, "thr_fail_step1", "thr_step1", default=0.5)
    thr_non_alt = _thr_get(thresholds, "thr_non_alt_step2a", "thr_step2a", default=0.5)
    thr_orig = _thr_get(thresholds, "thr_orig_step2b", "thr_step2b", default=0.5)
    thr_alt = _thr_get(thresholds, "thr_alt_step2c", "thr_step2c", default=0.5)

    steps: List[Tuple[str, str, Optional[int]]] = []

    # Step1: always
    if p_fail is not None and p_fail >= thr_fail:
        steps.append(("step1", "FAIL(미통과) 쪽 신호", None))
        steps.append(("stepf", "FAIL 세부결과 쪽 신호", None))
        return steps

    steps.append(("step1", "PASS(통과) 쪽 신호", None))

    if p_non_alt is None:
        return steps

    if p_non_alt >= thr_non_alt:
        steps.append(("step2a", "비대안(원안/수정) 쪽 신호", None))
        if p_orig is not None:
            steps.append(("step2b", "원안/수정 세부결과 쪽 신호", None))
    else:
        steps.append(("step2a", "대안계열(위원회 대안) 쪽 신호", None))
        if p_alt is not None:
            steps.append(("step2c", "대안 세부결과 쪽 신호", None))

    return steps


def _explain_feature(raw_name: str, value: Any, final_label: str) -> str:
    """피처별 1줄 해석(단정/인과 금지, 사용자 관점)."""
    name = _human_feature_name(raw_name)
    if _is_missing(value):
        return f"{name} 값이 비어 있어(결측) 모델이 참고할 근거가 줄었습니다."

    # 대표 템플릿(필요 시 추가)
    if raw_name == "elapsed_days":
        return "진행기간(경과일)은 ‘계류/처리 흐름’과 같이 움직이는 경우가 많아 모델이 민감하게 봅니다."
    if raw_name in ("proposer_count_est",):
        return "공동발의 규모는 ‘지지 기반/합의 가능성’의 간접 신호로 학습됐을 수 있습니다."
    if raw_name in ("proposer_kind", "ppsr_kind"):
        return "발의 주체(정부/의원/위원장)는 절차·논의 방식이 달라 결과 패턴이 달라질 수 있습니다."
    if raw_name in ("alternative_yn", "is_alternative"):
        return "대안 여부는 ‘위원회 대안 처리’ 경로와 직접 연결되는 핵심 메타입니다."
    if raw_name in ("amendment_type",):
        return "개정유형은 의안 성격을 요약해 주어, 유사 패턴 매칭에 자주 쓰입니다."
    if raw_name in ("budget_yn", "budget"):
        return "예산/비용추계 여부는 심사 난이도·절차에 영향을 주는 메타로 반영될 수 있습니다."
    if raw_name in ("party_at_term",):
        return "당시 정당/정치 환경은 법안 통과 패턴과 함께 관측될 수 있는 변수입니다."

    return f"{name} 값이 과거 데이터의 패턴과 함께 나타나, 예측에 참고 신호로 사용됩니다."


def _make_reason_sentence(name: str, value: Any, direction: str) -> str:
    val_s = "결측" if _is_missing(value) else str(_format_value(value))

    if val_s == "결측":
        return f"{name}가 비어 있어(결측) 근거가 약해질 수 있습니다."

    if direction in ("중요", "-"):
        return f"{name}({val_s})가 예측에 중요한 축으로 반영됐습니다."

    # direction: "예측 방향 ↑" / "예측 방향 ↓"
    return f"{name}({val_s})가 **{direction}** 신호로 크게 작용했습니다."


def _similarity_hint(similar_cases: Optional[Any], final_label: str) -> Tuple[Optional[str], Optional[str]]:
    if not similar_cases:
        return None, None

    rows = None
    if isinstance(similar_cases, list):
        rows = similar_cases
    elif isinstance(similar_cases, dict):
        rows = similar_cases.get("rows")
    if not isinstance(rows, list) or not rows:
        return None, None

    labs = [str(r.get("y_raw") or "") for r in rows[:10] if isinstance(r, dict)]
    labs = [x for x in labs if x]
    if len(labs) < 3:
        return None, None

    c = Counter(labs)
    top_lab, top_cnt = c.most_common(1)[0]
    n = len(labs)
    match = c.get(final_label, 0)

    if final_label and top_lab != final_label and top_cnt / n >= 0.5:
        return (
            f"유사사례 상위 {n}건에서는 ‘{top_lab}’가 더 자주 나타납니다({top_cnt}/{n}).",
            "유사사례 결과가 예측과 다르게 쏠려 있으면, 텍스트는 비슷하지만 절차/정치 맥락이 달랐을 수 있어요.",
        )
    if final_label and match <= n // 4:
        return (
            f"유사사례 상위 {n}건 중 예측({final_label})과 같은 결과가 적습니다({match}/{n}).",
            "요약(summary)을 더 정확히 넣고, 같은 위원회 필터를 켜면(가능하면) 더 ‘비슷한 맥락’의 사례를 찾습니다.",
        )
    return None, None


# -------------------------
# main
# -------------------------

def build_report_insights(
    *,
    rt: Any,
    X_one: pd.DataFrame,
    pred: Dict[str, Any],
    thresholds: Optional[Dict[str, Any]] = None,
    similar_cases: Optional[Any] = None,
    bill: Optional[Dict[str, Any]] = None,
    top_k_reasons: int = 3,
    top_k_features: int = 12,
) -> Dict[str, Any]:
    """보고서에 바로 넣을 insights dict 생성."""

    thresholds = thresholds or (
        pred.get("thresholds") if isinstance(pred.get("thresholds"), dict) else {}
    ) or {}

    final_label = str(pred.get("final_pred_8") or pred.get("final_pred") or "-")

    # confidence: threshold margin 휴리스틱
    p_fail = _get_prob(pred, "p_fail_step1", "p_fail")
    p_non_alt = _get_prob(pred, "p_non_alt")
    p_orig = _get_prob(pred, "p_orig")
    p_alt = _get_prob(pred, "p_alt")

    thr_fail = _thr_get(thresholds, "thr_fail_step1", "thr_step1", default=0.5)
    thr_non_alt = _thr_get(thresholds, "thr_non_alt_step2a", "thr_step2a", default=0.5)
    thr_orig = _thr_get(thresholds, "thr_orig_step2b", "thr_step2b", default=0.5)
    thr_alt = _thr_get(thresholds, "thr_alt_step2c", "thr_step2c", default=0.5)

    margins: List[float] = []
    if p_fail is not None:
        margins.append(abs(p_fail - thr_fail))
    if p_non_alt is not None:
        margins.append(abs(p_non_alt - thr_non_alt))
    if p_orig is not None:
        margins.append(abs(p_orig - thr_orig))
    if p_alt is not None:
        margins.append(abs(p_alt - thr_alt))
    min_margin = min(margins) if margins else None

    conf = "Medium"
    if min_margin is None:
        conf = "Medium"
    elif min_margin >= 0.20:
        conf = "High"
    elif min_margin >= 0.10:
        conf = "Medium"
    else:
        conf = "Low"

    # 1) SHAP/importance 수집 (경로 step만)
    steps = _decide_path(pred, thresholds)

    # contrib tuple: (feature, value, hint, mode)
    contribs_all: List[Tuple[str, float, str, str]] = []
    debug_steps: Dict[str, Any] = {}

    for step_key, hint, class_idx in steps:
        model = _pick_model(rt, step_key)
        if model is None:
            debug_steps[step_key] = {"mode": "missing_model"}
            continue

        mode = "shap"
        err: Optional[str] = None
        contribs: List[Tuple[str, float]] = []

        # 1) SHAP 시도
        try:
            contribs = _try_instance_shap_topk(model, X_one, class_idx, k=24)
        except Exception as e:
            err = str(e)
            contribs = []

        # 2) SHAP이 불가/빈 결과면 global importance 폴백
        if not contribs:
            mode = "global_importance"
            contribs = _try_global_importance_topk(model, k=24)

        debug_steps[step_key] = {"mode": mode, "n": len(contribs)}
        if err:
            debug_steps[step_key]["shap_err"] = err

        for f, c in contribs:
            if _is_banned_feature(str(f)):
                continue
            contribs_all.append((str(f), float(c), hint, mode))

    # 2) feature별 중요도 집계 + 임베딩 그룹핑
    # agg[name] = {raw, sum, max_abs, signed, sign_known, hint}
    agg: Dict[str, Dict[str, Any]] = {}

    for f, c, hint, mode in contribs_all:
        g = _group_feature(f) or f

        d = agg.get(g)
        if d is None:
            agg[g] = {
                "raw": f,
                "sum": 0.0,
                "max_abs": 0.0,
                "signed": 0.0,
                "sign_known": False,
                "hint": hint,
            }
            d = agg[g]

        d["sum"] += abs(c)

        # SHAP이면 sign 유지, global importance면 sign 모름
        if mode == "shap":
            if abs(c) > d["max_abs"]:
                d["max_abs"] = abs(c)
                d["signed"] = c
                d["sign_known"] = True
                d["raw"] = f
                d["hint"] = hint
        else:
            # global importance: sign 불명이라 signed는 0 유지
            if abs(c) > d["max_abs"]:
                d["max_abs"] = abs(c)
                d["raw"] = f
                d["hint"] = hint

    # 만약 모델 기여도 자체가 비어있다면, 최소 메타 피처로 폴백
    if not agg:
        fallback = [
            "elapsed_days",
            "committee_referral_days",
            "committee_meeting_days",
            "proposer_count_est",
            "ppsr_kind",
            "proposer_kind",
            "party_at_term",
            "amendment_type",
            "alternative_yn",
            "is_alternative",
            "budget_yn",
            "budget",
            "summary_embed",
            "billname_embed",
        ]
        for i, raw_name in enumerate(fallback):
            if _is_banned_feature(raw_name):
                continue
            agg[raw_name] = {
                "raw": raw_name,
                "sum": 1.0 / (1 + i),
                "max_abs": 1.0 / (1 + i),
                "signed": 0.0,
                "sign_known": False,
                "hint": "(fallback)",
            }

    ranked = sorted(agg.items(), key=lambda kv: float(kv[1].get("sum") or 0.0), reverse=True)

    # 3) top_features (표용)
    top_features_non_embed: List[Dict[str, Any]] = []
    top_features_embed: List[Dict[str, Any]] = []

    def _get_value(raw_name: str) -> Any:
        if raw_name in X_one.columns:
            return X_one.iloc[0].get(raw_name)
        if bill and raw_name in bill:
            return bill.get(raw_name)
        return None

    # 너무 임베딩이 도배되는 걸 피하기 위해 비임베딩을 먼저 채움
    for g, info in ranked[: max(top_k_features, 12) + 30]:
        raw_name = info.get("raw") if g not in ("billname_embed", "summary_embed", "text_embed") else g
        raw_name = str(raw_name)

        if _is_banned_feature(raw_name):
            continue

        disp = _human_feature_name(raw_name)
        importance = float(info.get("sum") or 0.0)
        signed = float(info.get("signed") or 0.0)
        sign_known = bool(info.get("sign_known"))

        value = _get_value(raw_name)

        if g in ("billname_embed", "summary_embed", "text_embed"):
            value = "(텍스트 패턴)"

        direction = "중요" if not sign_known else ("예측 방향 ↑" if signed >= 0 else "예측 방향 ↓")

        item = {
            "name": disp,
            "raw_name": raw_name,
            "value": None if _is_missing(value) else _format_value(value),
            "direction": direction,
            "importance": round(importance, 6),
            "desc": _explain_feature(raw_name, value, final_label),
        }

        if g in ("billname_embed", "summary_embed", "text_embed"):
            top_features_embed.append(item)
        else:
            top_features_non_embed.append(item)

        if len(top_features_non_embed) >= top_k_features:
            break

    top_features: List[Dict[str, Any]] = []
    top_features.extend(top_features_non_embed[:top_k_features])
    if len(top_features) < top_k_features:
        need = top_k_features - len(top_features)
        top_features.extend(top_features_embed[:need])

    # 4) top_reasons (요약용): top_features 상위 3개 기반
    reasons: List[str] = []
    for f in top_features[: max(top_k_reasons, 3)]:
        name = str(f.get("name") or "-")
        direction = str(f.get("direction") or "-")
        val = f.get("value")
        reasons.append(_make_reason_sentence(name, val, direction))
        if len(reasons) >= top_k_reasons:
            break
    while len(reasons) < top_k_reasons:
        reasons.append("(근거 생성 실패/부족)")

    # 5) process_summary (2~4문장, 사용자 관점)
    path_words: List[str] = []
    if final_label in FAIL_SET:
        path_words.append("미통과(FAIL) 경로로 분류")
    else:
        if p_fail is not None and p_fail < thr_fail:
            path_words.append("통과(PASS)로 먼저 분기")
        if p_non_alt is not None and p_non_alt >= thr_non_alt:
            path_words.append("비대안(원안/수정) 경로")
            if p_orig is not None:
                path_words.append("원안 쪽으로 수렴" if p_orig >= thr_orig else "수정 쪽으로 수렴")
        elif p_non_alt is not None:
            path_words.append("대안계열 경로")

    top_names = [str(x.get("name")) for x in top_features[:3] if x.get("name")]
    top_names = [n for n in top_names if n]

    summary_parts: List[str] = []
    if path_words:
        summary_parts.append(" → ".join(path_words) + f"하여 최종적으로 **{final_label}**로 예측했습니다.")
    else:
        summary_parts.append(f"모델이 입력 패턴을 종합해 **{final_label}**로 예측했습니다.")

    if top_names:
        summary_parts.append("특히 모델이 민감하게 본 요소는 " + ", ".join(top_names[:3]) + " 입니다.")

    s1, s2 = _similarity_hint(similar_cases, final_label)
    if s1:
        summary_parts.append(s1)

    process_summary = " ".join(summary_parts[:4]).strip()

    # 6) next_actions (사용자 관점)
    next_actions: List[str] = []

    missing_fields = [f.get("name") for f in top_features[:8] if _is_missing(f.get("value"))]
    if missing_fields:
        next_actions.append(
            "입력에서 중요한 항목(" + ", ".join([str(x) for x in missing_fields[:3]]) + ")이 비어 있어요. "
            "가능하면 원천 데이터로 값을 보강하면 근거(Top3/피처 표)가 더 정확해집니다."
        )
    else:
        next_actions.append(
            "이 결과는 입력 메타의 패턴에 민감합니다. ‘대안 여부/개정유형/예산수반’ 같은 핵심 메타가 실제와 일치하는지 한 번만 더 확인해 보세요."
        )

    if top_features:
        sim_targets = [f.get("name") for f in top_features[:3] if f.get("name")]
        next_actions.append(
            "시뮬레이션을 한다면 " + ", ".join([str(x) for x in sim_targets]) + " 값을 바꿔보는 게 가장 효율적입니다. "
            "(모델이 가장 크게 반응한 축이라 예측이 움직일 가능성이 큼)"
        )
    else:
        next_actions.append("시뮬레이션을 한다면 입력 피처(경과일/발의자수/대안 여부 등)를 바꿔보며 예측 변화를 확인해 보세요.")

    if s2:
        next_actions.append(s2)
    else:
        next_actions.append(
            "유사사례가 도움이 되려면 요약(summary)이 핵심입니다. 요약을 더 충실히 넣거나(핵심 조항/목표/대상), 같은 위원회 필터로 비교하면 해석이 쉬워집니다."
        )

    # Dedup & trim
    uniq: List[str] = []
    seen = set()
    for a in next_actions:
        a = str(a).strip()
        if not a or a in seen:
            continue
        uniq.append(a)
        seen.add(a)
        if len(uniq) >= 3:
            break
    while len(uniq) < 3:
        uniq.append("(추가 액션 생성이 제한됩니다)")

    return {
        "confidence": conf,
        "top_reasons": reasons[:top_k_reasons],
        "top_features": top_features,
        "process_summary": process_summary,
        "next_actions": uniq,
        "debug": {
            "min_margin": None if min_margin is None else float(min_margin),
            "steps": debug_steps,
        },
    }


__all__ = ["build_report_insights"]
