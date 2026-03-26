from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

from src.research_provider import (
    BaseResearchProvider,
    MockResearchProvider,
    SearchRequest,
)
from src.schemas import ResearchItem, ResearchOutput


STOPWORDS = {
    "the", "is", "a", "an", "of", "for", "about", "tell", "me", "recent",
    "what", "how", "their", "and", "at", "to", "in"
}


TOPIC_REQUIRED_HINTS = {
    "overview": {"overview"},
    "news": {"news"},
    "financials": {"financials"},
    "ceo": {"ceo"},
    "competitors": {"competitors"},
    "developments": {"developments"},
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> Set[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {t for t in tokens if t not in STOPWORDS}


def _extract_topic_from_state(state: dict) -> str:
    topic = state.get("last_topic")
    if not topic:
        raise ValueError("last_topic is required before running ResearchAgent")
    return topic


def _extract_company_from_state(state: dict) -> str:
    company = state.get("active_company")
    if not company:
        raise ValueError("active_company is required before running ResearchAgent")
    return company


def _build_gaps(topic: str, findings: List[dict]) -> List[str]:
    gaps = []
    expected_categories = TOPIC_REQUIRED_HINTS.get(topic, {topic})

    found_categories = {item.get("category", "") for item in findings}

    if not findings:
        gaps.append(f"No research findings retrieved for topic '{topic}'.")
        return gaps

    if not (expected_categories & found_categories):
        gaps.append(f"Missing topic-specific evidence for '{topic}'.")

    if len(findings) < 1:
        gaps.append("Insufficient number of findings.")

    return gaps


def _compute_rule_confidence(topic: str, findings: List[dict], gaps: List[str]) -> float:
    """
    Deterministic confidence from:
    - topic match
    - number of findings
    - gap severity
    """
    if not findings:
        return 1.0

    score = 0.0

    found_categories = {item.get("category", "") for item in findings}
    expected_categories = TOPIC_REQUIRED_HINTS.get(topic, {topic})

    # topic coverage
    if expected_categories & found_categories:
        score += 4.0
    elif "overview" in found_categories:
        score += 2.0

    # quantity
    if len(findings) >= 1:
        score += 2.0
    if len(findings) >= 2:
        score += 1.0
    if len(findings) >= 3:
        score += 1.0

    # gap penalty
    score -= min(len(gaps) * 2.0, 4.0)

    # summary quality proxy
    non_empty_summaries = sum(1 for item in findings if item.get("summary"))
    if non_empty_summaries == len(findings):
        score += 2.0

    return max(0.0, min(score, 10.0))


class HeuristicLLMConfidenceScorer:
    """
    Honest placeholder for V1 local testing.
    Later you can replace this with a real LLM-backed scorer
    without changing ResearchAgent.
    """

    def score(
        self,
        resolved_query: str,
        topic: str,
        findings: List[dict],
        gaps: List[str],
    ) -> float:
        if not findings:
            return 1.0

        query_tokens = _tokenize(resolved_query)
        corpus = " ".join(
            f"{item.get('title', '')} {item.get('summary', '')}" for item in findings
        )
        corpus_tokens = _tokenize(corpus)

        overlap = len(query_tokens & corpus_tokens)
        coverage_ratio = overlap / max(len(query_tokens), 1)

        score = 0.0

        # semantic-ish overlap proxy
        if coverage_ratio >= 0.6:
            score += 5.0
        elif coverage_ratio >= 0.3:
            score += 3.5
        else:
            score += 2.0

        # topic presence
        found_categories = {item.get("category", "") for item in findings}
        if topic in found_categories:
            score += 2.5
        elif "overview" in found_categories:
            score += 1.0

        # completeness penalty
        if not gaps:
            score += 2.5
        else:
            score -= min(len(gaps) * 1.5, 3.0)

        return max(0.0, min(score, 10.0))


class ResearchAgent:
    def __init__(
        self,
        provider: Optional[BaseResearchProvider] = None,
        llm_confidence_scorer: Optional[HeuristicLLMConfidenceScorer] = None,
    ):
        self.provider = provider or MockResearchProvider()
        self.llm_confidence_scorer = llm_confidence_scorer or HeuristicLLMConfidenceScorer()

    def run(self, state: dict) -> ResearchOutput:
        company = _extract_company_from_state(state)
        topic = _extract_topic_from_state(state)
        resolved_query = state.get("resolved_query") or f"Tell me about {company}"

        # If attempt_count is 0 in state, research invocation becomes attempt 1
        current_attempt = int(state.get("attempt_count", 0)) + 1
        validator_feedback = state.get("validation_feedback")

        request = SearchRequest(
            company=company,
            topic=topic,
            resolved_query=resolved_query,
            attempt_count=current_attempt,
            validator_feedback=validator_feedback,
        )

        result = self.provider.search(request)
        findings = result.findings
        gaps = _build_gaps(topic, findings)

        rule_confidence_score = _compute_rule_confidence(topic, findings, gaps)
        llm_confidence_score = self.llm_confidence_scorer.score(
            resolved_query=resolved_query,
            topic=topic,
            findings=findings,
            gaps=gaps,
        )

        confidence_gate_passed = (
            rule_confidence_score >= 6.0 and llm_confidence_score >= 6.0
        )

        research_items = [
            ResearchItem(
                category=item["category"],
                title=item["title"],
                summary=item["summary"],
                source_hint=item.get("source_hint"),
            )
            for item in findings
        ]

        # Add retrieval notes as gaps only when there are actual content problems? No.
        # Keep them out of user-facing research gaps. You can keep them in state later if needed.

        return ResearchOutput(
            company=company,
            topic=topic,
            resolved_query=resolved_query,
            findings=research_items,
            gaps=gaps,
            rule_confidence_score=rule_confidence_score,
            llm_confidence_score=llm_confidence_score,
            confidence_gate_passed=confidence_gate_passed,
        )