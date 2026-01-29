# backend/model_runtime.py
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


PASS_SET = {"원안가결", "수정가결", "대안반영폐기", "수정안반영폐기"}
FAIL_SET = {"부결", "폐기", "철회", "임기만료폐기"}

ALT_SET = {"대안반영폐기", "수정안반영폐기"}
NON_ALT_SET = {"원안가결", "수정가결"}


# training script와 동일한 강제 cat 목록(있으면 cat으로 취급)
FORCE_CAT = [
    "ntr_div",
    "election_type",
    "party_at_term",
    "district_at_term",
    "is_negotiation_group_party",
    "has_committee",
    "committee_at_term",
    "curr_committee_code",
    "curr_committee_name",
    "proposer_kind",
    "pass_gubn",
    "amendment_type",
    "is_alternative",
    "budget",
    "proposer_name",
    # ✅ 추가: 모델에서 categorical로 학습된 컬럼은 런타임에서도 무조건 categorical(string)로 맞춘다
    "social_issue_yn",
    "summary_embedding",
]

# ✅ 숫자여야 하는데 DB에서 object/string로 섞여 들어오는 컬럼 강제 numeric 처리
FORCE_NUM = {
    "social_issue_score",
    "ai_probability",
    # 필요하면 추가:
    # "social_issue_count",
}

EXCLUDE_ALWAYS_NULL_COLS = {
    "competent_ministry",
    "committee_referral_count",
    "committee_referral_days",
    "committee_meeting_count",
    "committee_meeting_days",
}
EXCLUDE_FEATURE_COLS = {"proc_stage_cd"}
EXCLUDE_TERM_COLS = {"term"}
EXCLUDE_ENGINEERED_COLS = {"lwcmt_meeting_intensity"}

RESULT_COL = "general_result"


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        return float(x)
    except Exception:
        return None


def _normalize_result(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip().replace("\u00a0", " ")
    return " ".join(s.split())


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 날짜 파생: propose_dt -> year/month/dow
    if "propose_dt" in df.columns:
        df["propose_dt"] = pd.to_datetime(df["propose_dt"], errors="coerce")
        df["propose_year"] = df["propose_dt"].dt.year
        df["propose_month"] = df["propose_dt"].dt.month
        df["propose_dow"] = df["propose_dt"].dt.dayofweek
    else:
        for c in ["propose_year", "propose_month", "propose_dow"]:
            if c not in df.columns:
                df[c] = np.nan

    # 숫자형 정리
    num_cols = ["elapsed_days", "proposer_count_est", "lwcmt_meeting_count"]
    for c in num_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # training에서 제거했던 lwcmt_meeting_intensity는 만들지 않음(안전)
    if "lwcmt_meeting_intensity" in df.columns:
        df = df.drop(columns=["lwcmt_meeting_intensity"], errors="ignore")

    df["log_proposer"] = np.log1p(df["proposer_count_est"].fillna(0))
    df["log_elapsed"] = np.log1p(df["elapsed_days"].fillna(0))

    return df


def _expected_features(model: CatBoostClassifier) -> List[str]:
    """
    저장된 CatBoost 모델이 기대하는 feature name list를 최대한 안정적으로 가져옴.
    """
    for attr in ["feature_names_", "feature_names"]:
        if hasattr(model, attr):
            v = getattr(model, attr)
            if isinstance(v, list) and len(v) > 0:
                return v

    if hasattr(model, "get_feature_names"):
        try:
            v = model.get_feature_names()
            if isinstance(v, list) and len(v) > 0:
                return v
        except Exception:
            pass

    return []


def _build_feature_columns_from_expected(expected: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    expected: model이 기대하는 피처명 리스트
    - text: bill_name, summary
    - cat: FORCE_CAT + (preprocess에서 object면 자동 cat 처리)
    """
    features = list(expected)

    text_cols = []
    for t in ["bill_name", "summary"]:
        if t in features:
            text_cols.append(t)

    cat_cols = []
    for c in FORCE_CAT:
        if c in features and c not in text_cols:
            cat_cols.append(c)

    return features, cat_cols, text_cols


def preprocess_X(df: pd.DataFrame, features: List[str], cat_cols: List[str], text_cols: List[str]) -> pd.DataFrame:
    """
    ✅ 핵심 수정:
    - social_issue_score 같은 numeric 컬럼이 DB에서 object/string로 들어오면,
      기존 로직은 object => cat_features 로 마킹해서 CatBoostError 발생.
    - FORCE_NUM은 무조건 numeric으로 강제하고 cat_cols에서 제거.
    - 기타 object/string 컬럼은 "대부분 숫자로 파싱되면" numeric, 아니면 cat.
    """
    X = df.copy()

    # 필요한 컬럼 없으면 생성
    for c in features:
        if c not in X.columns:
            X[c] = np.nan

    X = X[features].copy()

    # text: 빈문자열
    for c in text_cols:
        if c in X.columns:
            X[c] = X[c].astype("string").fillna("")

    # 1) 강제 numeric 처리 (가장 중요)
    for c in FORCE_NUM:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
            if c in cat_cols:
                cat_cols.remove(c)

    # 2) 강제 cat 목록은 cat으로 고정
    forced_cat = set(cat_cols)
    for c in list(forced_cat):
        if c in X.columns and c not in text_cols:
            X[c] = X[c].astype("string").fillna("NA")

    # 3) object/string 컬럼 처리: "대부분 숫자"면 numeric, 아니면 cat
    for c in X.columns:
        if c in text_cols:
            continue
        if c in forced_cat:
            continue
        if c in FORCE_NUM:
            continue
        if pd.api.types.is_numeric_dtype(X[c]):
            continue

        if pd.api.types.is_object_dtype(X[c]) or pd.api.types.is_string_dtype(X[c]):
            s = X[c].astype("string")
            num = pd.to_numeric(s, errors="coerce")

            non_empty = s.notna() & (s.str.strip() != "")
            ratio = float(num[non_empty].notna().mean()) if non_empty.any() else 0.0

            if ratio >= 0.8:
                X[c] = num
            else:
                X[c] = s.fillna("NA")
                if c not in cat_cols:
                    cat_cols.append(c)

    # 4) 최종 cat_cols 정리: numeric으로 확정된 건 cat에서 제거
    cleaned = []
    for c in cat_cols:
        if c in X.columns and c not in text_cols and not pd.api.types.is_numeric_dtype(X[c]):
            cleaned.append(c)
    cat_cols[:] = cleaned

    # 5) 나머지 numeric 정리 + fill
    num_cols = [c for c in features if c not in set(cat_cols) and c not in set(text_cols)]
    for c in num_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    if len(num_cols) > 0:
        X[num_cols] = X[num_cols].fillna(-1)

    return X


def make_pool(X: pd.DataFrame, y, cat_cols: List[str], text_cols: List[str]) -> Pool:
    cat_idx = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]
    text_idx = [X.columns.get_loc(c) for c in text_cols if c in X.columns]
    if y is None:
        return Pool(X, cat_features=cat_idx, text_features=text_idx)
    return Pool(X, y, cat_features=cat_idx, text_features=text_idx)


@dataclass
class ModelPaths:
    step1: str
    step2a: str
    step2b: str
    step2c: str
    stepF: str
    thresholds_json: str


def default_model_paths(models_dir: str = "models") -> ModelPaths:
    """
    models/
      cbm_step1_....cbm
      cbm_step2a_....cbm
      cbm_step2b_....cbm
      cbm_step2c_....cbm
      cbm_stepF_....cbm
      thresholds_8class_....json
    """
    d = Path(models_dir)
    return ModelPaths(
        step1=str(d / "cbm_step1_passfail_billname_embed_noSummary_noProc_noTerm_useOrd_noLwcmtIntensity.cbm"),
        step2a=str(d / "cbm_step2a_nonalt_vs_alt_billname_embed_noSummary_noProc_noTerm_useOrd_noLwcmtIntensity.cbm"),
        step2b=str(d / "cbm_step2b_orig_vs_amend_billname_embed_noSummary_noProc_noTerm_useOrd_noLwcmtIntensity.cbm"),
        step2c=str(d / "cbm_step2c_alt_detail_billname_embed_noSummary_noProc_noTerm_useOrd_noLwcmtIntensity.cbm"),
        stepF=str(d / "cbm_stepF_fail4_billname_embed_noSummary_noProc_noTerm_useOrd_noLwcmtIntensity.cbm"),
        thresholds_json=str(
            d / "thresholds_8class_billname_embed_noSummary_noProc_noTerm_useOrd_noLwcmtIntensity.json"
        ),
    )


class ModelRuntime:
    """
    - 저장된 5개 CatBoost 모델(.cbm) + thresholds json 로드
    - 입력: bill feature dict or DataFrame
    - 출력: p_fail_step1, p_non_alt, p_orig, p_alt, final_pred_8
    """

    def __init__(self, paths: ModelPaths, model_version: str = "v123456789"):
        self.paths = paths
        self.model_version = model_version

        self.step1 = CatBoostClassifier()
        self.step2a = CatBoostClassifier()
        self.step2b = CatBoostClassifier()
        self.step2c = CatBoostClassifier()
        self.stepF = CatBoostClassifier()

        self.thresholds: Dict[str, Any] = {}
        self.model_info: Dict[str, Any] = {
            "model_version": model_version,
            "step1_path": paths.step1,
            "step2a_path": paths.step2a,
            "step2b_path": paths.step2b,
            "step2c_path": paths.step2c,
            "stepF_path": paths.stepF,
            "thresholds_json": paths.thresholds_json,
        }

        self._load()

    def _load(self):
        for k, p in self.model_info.items():
            if k.endswith("_path") or k == "thresholds_json":
                if not Path(p).exists():
                    raise FileNotFoundError(f"[ModelRuntime] file not found: {p}")

        self.step1.load_model(self.paths.step1)
        self.step2a.load_model(self.paths.step2a)
        self.step2b.load_model(self.paths.step2b)
        self.step2c.load_model(self.paths.step2c)
        self.stepF.load_model(self.paths.stepF)

        with open(self.paths.thresholds_json, "r", encoding="utf-8") as f:
            self.thresholds = json.load(f)

    def _predict_step1_pfail(self, df: pd.DataFrame) -> np.ndarray:
        expected = _expected_features(self.step1)
        if not expected:
            expected = list(df.columns)

        features, cat_cols, text_cols = _build_feature_columns_from_expected(expected)
        df2 = add_engineered_features(df)
        X = preprocess_X(df2, features, cat_cols, text_cols)
        pool = make_pool(X, None, cat_cols, text_cols)
        p_fail = self.step1.predict_proba(pool)[:, 1]
        return p_fail

    def predict_8class_df(self, df_input: pd.DataFrame) -> pd.DataFrame:
        df = df_input.copy()
        df = add_engineered_features(df)

        p_fail = self._predict_step1_pfail(df)
        out = df.copy()
        out["p_fail_step1"] = p_fail

        thr_fail = _safe_float(self.thresholds.get("thr_fail_step1"))
        thr_non_alt = _safe_float(self.thresholds.get("thr_non_alt_step2a"))
        thr_orig = _safe_float(self.thresholds.get("thr_orig_step2b"))
        thr_alt = _safe_float(self.thresholds.get("thr_alt_step2c"))

        out["p_non_alt"] = np.nan
        out["p_orig"] = np.nan
        out["p_alt"] = np.nan
        out["final_pred_8"] = "폐기"

        if thr_fail is None:
            thr_fail = 0.5

        mask_pass = out["p_fail_step1"] < thr_fail
        mask_fail = ~mask_pass

        # FAIL: stepF 4-class
        if mask_fail.any():
            expected = _expected_features(self.stepF)
            if not expected:
                expected = list(out.columns)
            features, cat_cols, text_cols = _build_feature_columns_from_expected(expected)

            df_fail = out.loc[mask_fail].copy()
            Xf = preprocess_X(df_fail, features, cat_cols, text_cols)
            poolf = make_pool(Xf, None, cat_cols, text_cols)
            pred_fail4 = self.stepF.predict(poolf).ravel()
            out.loc[mask_fail, "final_pred_8"] = pred_fail4

        # PASS: step2a -> step2b/step2c
        if mask_pass.any():
            if thr_non_alt is None:
                thr_non_alt = 0.5
            if thr_orig is None:
                thr_orig = 0.5
            if thr_alt is None:
                thr_alt = 0.5

            # Step2A
            expected = _expected_features(self.step2a)
            if not expected:
                expected = list(out.columns)
            features, cat_cols, text_cols = _build_feature_columns_from_expected(expected)

            df_pass = out.loc[mask_pass].copy()
            X2a = preprocess_X(df_pass, features, cat_cols, text_cols)
            pool2a = make_pool(X2a, None, cat_cols, text_cols)
            p_non_alt = self.step2a.predict_proba(pool2a)[:, 1]
            out.loc[mask_pass, "p_non_alt"] = p_non_alt

            mask_non_alt = mask_pass & (out["p_non_alt"] >= thr_non_alt)
            mask_alt = mask_pass & ~mask_non_alt

            # Step2B (비대안)
            if mask_non_alt.any():
                expected = _expected_features(self.step2b)
                if not expected:
                    expected = list(out.columns)
                features, cat_cols, text_cols = _build_feature_columns_from_expected(expected)

                df_non_alt = out.loc[mask_non_alt].copy()
                X2b = preprocess_X(df_non_alt, features, cat_cols, text_cols)
                pool2b = make_pool(X2b, None, cat_cols, text_cols)
                p_orig = self.step2b.predict_proba(pool2b)[:, 1]
                out.loc[mask_non_alt, "p_orig"] = p_orig
                out.loc[mask_non_alt, "final_pred_8"] = np.where(p_orig >= thr_orig, "원안가결", "수정가결")

            # Step2C (대안계열)
            if mask_alt.any():
                expected = _expected_features(self.step2c)
                if not expected:
                    expected = list(out.columns)
                features, cat_cols, text_cols = _build_feature_columns_from_expected(expected)

                df_alt = out.loc[mask_alt].copy()
                X2c = preprocess_X(df_alt, features, cat_cols, text_cols)
                pool2c = make_pool(X2c, None, cat_cols, text_cols)
                p_alt = self.step2c.predict_proba(pool2c)[:, 1]
                out.loc[mask_alt, "p_alt"] = p_alt
                out.loc[mask_alt, "final_pred_8"] = np.where(p_alt >= thr_alt, "수정안반영폐기", "대안반영폐기")

        return out

    def predict_8class_one(self, bill_features: Dict[str, Any]) -> Dict[str, Any]:
        df = pd.DataFrame([bill_features])
        pred_df = self.predict_8class_df(df)
        row = pred_df.iloc[0].to_dict()

        return {
            "p_fail_step1": _safe_float(row.get("p_fail_step1")),
            "p_non_alt": _safe_float(row.get("p_non_alt")),
            "p_orig": _safe_float(row.get("p_orig")),
            "p_alt": _safe_float(row.get("p_alt")),
            "final_pred_8": row.get("final_pred_8"),
            "model_version": self.model_version,
            "thresholds": {
                "thr_fail_step1": _safe_float(self.thresholds.get("thr_fail_step1")),
                "thr_non_alt_step2a": _safe_float(self.thresholds.get("thr_non_alt_step2a")),
                "thr_orig_step2b": _safe_float(self.thresholds.get("thr_orig_step2b")),
                "thr_alt_step2c": _safe_float(self.thresholds.get("thr_alt_step2c")),
            },
        }


def load_runtime(models_dir: str = "models", model_version: str = "v123456789") -> ModelRuntime:
    paths = default_model_paths(models_dir=models_dir)
    return ModelRuntime(paths=paths, model_version=model_version)


if __name__ == "__main__":
    rt = load_runtime(models_dir="models", model_version="v123456789")
    sample = {
        "bill_name": "테스트 의안",
        "summary": "이 의안은 테스트를 위한 요약입니다.",
        "elapsed_days": 30,
        "proposer_count_est": 10,
        "lwcmt_meeting_count": 1,
        "social_issue_score": "0.73",  # string으로 들어와도 이제 안전
        "social_issue_yn": 1,          # ✅ 0/1로 들어와도 categorical(string)로 강제됨
    }
    print(rt.predict_8class_one(sample))
