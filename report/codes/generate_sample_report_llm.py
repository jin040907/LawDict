"""
Generate a sample report (Markdown).
- Uses report_builder_with_llm to render the base report.
- Optionally enriches some sections via LLM (section-wise prompts), but has safe fallback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from backend.report_builder_with_llm import build_report_markdown
from backend.llm_report_enricher import enrich_for_report


def load_input_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def _merge_evidence_hint(evidence: Any, pred: Dict[str, Any]) -> Dict[str, Any]:
    """
    LLM에 '근거 후보(특히 top_features)'를 최대한 전달:
    - pred.insights (report_insights에서 만든 값) 우선
    - evidence가 있으면 그 위에 merge
    """
    hint: Dict[str, Any] = {}
    ins = pred.get("insights")
    if isinstance(ins, dict):
        hint.update(ins)
    if isinstance(evidence, dict):
        hint.update(evidence)
    return hint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="bill/pred/... payload JSON 경로")
    ap.add_argument("--out", default="sample_report_llm.md", help="출력 md 파일")
    ap.add_argument("--model", default=None, help="OpenAI model override (예: gpt-5.2)")
    ap.add_argument("--no_llm", action="store_true", help="LLM enrich 없이 바로 md 생성")
    args = ap.parse_args()

    payload = load_input_json(args.input)

    bill = payload.get("bill") or {}
    pred = payload.get("pred") or payload.get("prediction") or {}
    evidence = payload.get("evidence")
    similar_cases = payload.get("similar_cases")
    timeline = payload.get("timeline")
    model_info = payload.get("model_info")

    evidence_hint = _merge_evidence_hint(evidence, pred)

    evidence2 = evidence_hint
    similar_cases2 = similar_cases

    if not args.no_llm:
        try:
            enriched = enrich_for_report(
                bill=bill,
                pred=pred,
                evidence=evidence_hint,
                similar_cases=similar_cases,
                timeline=timeline,
                model_info=model_info,
                model=args.model,
            )
            if isinstance(enriched, dict):
                evidence2 = enriched.get("evidence", evidence_hint)
                similar_cases2 = enriched.get("similar_cases", similar_cases)
        except Exception as e:
            print(f"// [WARN] LLM enrich failed: {type(e).__name__}: {e}")
            # fallback: LLM 없이도 md는 생성

    md = build_report_markdown(
        bill=bill,
        pred=pred,
        evidence=evidence2,
        similar_cases=similar_cases2,
        timeline=timeline,
        model_info=model_info,
    )

    Path(args.out).write_text(md, encoding="utf-8")
    print(f"// ✅ wrote: {args.out}")


if __name__ == "__main__":
    main()
