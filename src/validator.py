from __future__ import annotations

from typing import List, Set

from src.schemas import ValidationOutput


TOPIC_EXPECTED_CATEGORIES = {
    "overview": {"overview"},
    "news": {"news"},
    "financials": {"financials"},
    "ceo": {"ceo"},
    "competitors": {"competitors"},
    "developments": {"developments"},
}


def _extract_research_payload(state: dict) -> dict:
    research_output = state.get("research_output")
    if not research_output:
        raise ValueError("research_output is required before running ValidatorAgent")
    return research_output


def _get_topic(state: dict, research_output: dict) -> str:
    topic = state.get("last_topic") or research_output.get("topic")
    if not topic:
        raise ValueError("last_topic/topic is required before running ValidatorAgent")
    return topic


def _get_resolved_query(state: dict, research_output: dict) -> str:
    resolved_query = state.get("resolved_query") or research_output.get("resolved_query")
    if not resolved_query:
        raise ValueError("resolved_query is required before running ValidatorAgent")
    return resolved_query


def _categories_from_findings(findings: List[dict]) -> Set[str]:
    return {item.get("category", "") for item in findings if item.get("category")}


def _non_empty_findings(findings: List[dict]) -> int:
    count = 0
    for item in findings:
        title_ok = bool(item.get("title"))
        summary_ok = bool(item.get("summary"))
        if title_ok and summary_ok:
            count += 1
    return count


def _relevance_check(topic: str, findings: List[dict]) -> bool:
    categories = _categories_from_findings(findings)
    expected = TOPIC_EXPECTED_CATEGORIES.get(topic, {topic})

    # direct topic match is best
    if expected & categories:
        return True

    # overview is acceptable only for overview topic
    if topic == "overview" and "overview" in categories:
        return True

    return False


def _completeness_check(topic: str, findings: List[dict], research_gaps: List[str]) -> bool:
    if not findings:
        return False

    if _non_empty_findings(findings) == 0:
        return False

    if research_gaps:
        return False

    if topic == "overview":
        return _non_empty_findings(findings) >= 1

    # for non-overview topics require at least one direct-topic finding
    return _relevance_check(topic, findings)


def _safe_to_answer(topic: str, findings: List[dict]) -> bool:
    # grounded-only safety: only answer topic if topic evidence exists
    return _relevance_check(topic, findings)


def _build_feedback(
    topic: str,
    resolved_query: str,
    findings: List[dict],
    research_gaps: List[str],
    relevance_ok: bool,
    completeness_ok: bool,
    safe_ok: bool,
) -> str:
    messages = []

    if not findings:
        messages.append(
            f"No findings were retrieved for topic '{topic}'. Try a direct {topic} lookup for the company."
        )
        return " ".join(messages)

    if not relevance_ok:
        messages.append(
            f"Current findings are not relevant enough to answer '{resolved_query}'."
        )
        messages.append(
            f"Need topic-specific evidence for '{topic}'."
        )

    if not completeness_ok:
        if research_gaps:
            messages.append(
                f"Research gaps detected: {'; '.join(research_gaps)}."
            )
        else:
            messages.append(
                f"Research is incomplete for topic '{topic}'."
            )

    if not safe_ok:
        messages.append(
            "Do not answer from unrelated categories. Retrieve direct evidence for the requested topic only."
        )

    if topic == "financials":
        messages.append(
            "Try broader financial or stock-related evidence, then fall back to overview only as supporting context."
        )
    elif topic == "news":
        messages.append(
            "Try recent news-specific evidence rather than generic company overview."
        )
    elif topic == "ceo":
        messages.append(
            "Retrieve leadership or CEO-specific evidence."
        )
    elif topic == "competitors":
        messages.append(
            "Retrieve competitor or rivalry-specific evidence."
        )
    elif topic == "developments":
        messages.append(
            "Retrieve recent developments, launches, roadmap, or acquisition-related evidence."
        )
    elif topic == "overview":
        messages.append(
            "Retrieve a direct company overview."
        )

    return " ".join(messages)


class ValidatorAgent:
    def run(self, state: dict) -> ValidationOutput:
        research_output = _extract_research_payload(state)
        topic = _get_topic(state, research_output)
        resolved_query = _get_resolved_query(state, research_output)

        findings = research_output.get("findings", [])
        research_gaps = research_output.get("gaps", [])

        relevance_ok = _relevance_check(topic, findings)
        completeness_ok = _completeness_check(topic, findings, research_gaps)
        safe_ok = _safe_to_answer(topic, findings)

        if relevance_ok and completeness_ok and safe_ok:
            return ValidationOutput(
                validation_result="sufficient",
                validation_feedback=(
                    f"Research is sufficient for '{resolved_query}'."
                ),
            )

        feedback = _build_feedback(
            topic=topic,
            resolved_query=resolved_query,
            findings=findings,
            research_gaps=research_gaps,
            relevance_ok=relevance_ok,
            completeness_ok=completeness_ok,
            safe_ok=safe_ok,
        )

        return ValidationOutput(
            validation_result="insufficient",
            validation_feedback=feedback,
        )