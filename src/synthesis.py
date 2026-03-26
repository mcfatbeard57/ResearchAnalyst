from __future__ import annotations

from typing import Dict, List, Tuple

from src.schemas import SynthesisOutput


TOPIC_LABELS = {
    "overview": "overview",
    "news": "recent news",
    "financials": "financials",
    "ceo": "CEO / leadership",
    "competitors": "competitors",
    "developments": "recent developments",
}


def _get_research_output(state: dict) -> dict:
    research_output = state.get("research_output")
    if not research_output:
        raise ValueError("research_output is required before running SynthesisAgent")
    return research_output


def _get_company(state: dict, research_output: dict) -> str:
    company = state.get("active_company") or research_output.get("company")
    if not company:
        raise ValueError("active_company/company is required before running SynthesisAgent")
    return company


def _get_topic(state: dict, research_output: dict) -> str:
    topic = state.get("last_topic") or research_output.get("topic")
    if not topic:
        raise ValueError("last_topic/topic is required before running SynthesisAgent")
    return topic


def _normalize_findings(research_output: dict) -> List[dict]:
    findings = research_output.get("findings", [])
    if not isinstance(findings, list):
        return []
    return findings


def _split_findings(topic: str, findings: List[dict]) -> Tuple[List[dict], List[dict]]:
    direct = []
    supporting = []

    for item in findings:
        category = item.get("category", "")
        if category == topic:
            direct.append(item)
        else:
            supporting.append(item)

    return direct, supporting


def _is_incomplete(state: dict, research_output: dict, direct_findings: List[dict]) -> bool:
    validation_result = state.get("validation_result")
    attempt_count = int(state.get("attempt_count", 0))
    gaps = research_output.get("gaps", [])
    confidence_gate_passed = research_output.get("confidence_gate_passed")

    if validation_result == "insufficient":
        return True
    if attempt_count >= 3:
        return True
    if gaps:
        return True
    if confidence_gate_passed is False:
        return True
    if len(direct_findings) == 0:
        return True

    return False


def _clean_summary(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    return text.rstrip(".")


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    deduped = []
    seen = set()
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            deduped.append(item.strip())
            seen.add(key)
    return deduped


def _render_direct_points(direct_findings: List[dict]) -> List[str]:
    points = []
    for item in direct_findings:
        summary = _clean_summary(item.get("summary", ""))
        title = (item.get("title") or "").strip()

        if summary and title:
            points.append(f"- {title}: {summary}.")
        elif summary:
            points.append(f"- {summary}.")
        elif title:
            points.append(f"- {title}.")
    return _dedupe_preserve_order(points)


def _render_supporting_points(supporting_findings: List[dict]) -> List[str]:
    points = []
    for item in supporting_findings:
        summary = _clean_summary(item.get("summary", ""))
        category = (item.get("category") or "other").strip()

        if summary:
            points.append(f"- ({category}) {summary}.")
    return _dedupe_preserve_order(points)


def _build_complete_response(
    company: str,
    topic: str,
    direct_findings: List[dict],
) -> str:
    topic_label = TOPIC_LABELS.get(topic, topic)
    lines = [f"Here is a grounded summary of {company}'s {topic_label}:"]

    direct_points = _render_direct_points(direct_findings)
    if direct_points:
        lines.extend(direct_points)
    else:
        lines.append(f"- I do not have direct retrieved information for {company}'s {topic_label}.")

    return "\n".join(lines)


def _build_incomplete_response(
    company: str,
    topic: str,
    direct_findings: List[dict],
    supporting_findings: List[dict],
    validation_feedback: str | None,
) -> str:
    topic_label = TOPIC_LABELS.get(topic, topic)

    lines = [
        "I could not gather enough complete information to fully answer this request.",
        "Here is the best available summary based on the retrieved research.",
        "",
    ]

    if direct_findings:
        lines.append(f"Available information for {company}'s {topic_label}:")
        lines.extend(_render_direct_points(direct_findings))
    else:
        lines.append(
            f"I do not have enough topic-specific retrieved information for {company}'s {topic_label}."
        )

    if supporting_findings:
        lines.append("")
        lines.append("Supporting context from other retrieved categories:")
        lines.extend(_render_supporting_points(supporting_findings))

    if validation_feedback:
        lines.append("")
        lines.append(f"Known limitation: {validation_feedback}")

    return "\n".join(lines)


class SynthesisAgent:
    def run(self, state: dict) -> SynthesisOutput:
        research_output = _get_research_output(state)
        company = _get_company(state, research_output)
        topic = _get_topic(state, research_output)
        findings = _normalize_findings(research_output)
        validation_feedback = state.get("validation_feedback")

        direct_findings, supporting_findings = _split_findings(topic, findings)
        incomplete = _is_incomplete(state, research_output, direct_findings)

        if incomplete:
            final_response = _build_incomplete_response(
                company=company,
                topic=topic,
                direct_findings=direct_findings,
                supporting_findings=supporting_findings,
                validation_feedback=validation_feedback,
            )
        else:
            final_response = _build_complete_response(
                company=company,
                topic=topic,
                direct_findings=direct_findings,
            )

        return SynthesisOutput(final_response=final_response)