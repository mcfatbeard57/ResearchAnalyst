from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.mock_data import MOCK_COMPANY_DATA


@dataclass
class SearchRequest:
    company: str
    topic: str
    resolved_query: str
    attempt_count: int = 1
    validator_feedback: Optional[str] = None


@dataclass
class SearchResult:
    company: str
    topic: str
    findings: List[Dict] = field(default_factory=list)
    retrieval_notes: List[str] = field(default_factory=list)


class BaseResearchProvider(ABC):
    @abstractmethod
    def search(self, request: SearchRequest) -> SearchResult:
        raise NotImplementedError


class MockResearchProvider(BaseResearchProvider):
    """
    Attempt behavior:
    - Attempt 1: exact topic only
    - Attempt 2: exact topic + overview
    - Attempt 3: exact topic + overview + targeted fallback from validator feedback
    """

    def __init__(self, data: Optional[Dict] = None):
        self.data = data or MOCK_COMPANY_DATA

    def search(self, request: SearchRequest) -> SearchResult:
        company_data = self.data.get(request.company, {})
        findings: List[Dict] = []
        retrieval_notes: List[str] = []

        # Attempt 1: exact topic
        exact_items = company_data.get(request.topic, [])
        findings.extend(exact_items)
        retrieval_notes.append(f"attempt_{request.attempt_count}: exact_topic_lookup")

        # Attempt 2+: include overview context if different topic
        if request.attempt_count >= 2 and request.topic != "overview":
            overview_items = company_data.get("overview", [])
            findings.extend(overview_items)
            retrieval_notes.append("broadening_with_overview")

        # Attempt 3: targeted fallback from validator feedback
        if request.attempt_count >= 3 and request.validator_feedback:
            lowered = request.validator_feedback.lower()

            candidate_topics = [
                "overview",
                "news",
                "financials",
                "ceo",
                "competitors",
                "developments",
            ]

            for topic in candidate_topics:
                if topic in lowered and topic != request.topic:
                    findings.extend(company_data.get(topic, []))
                    retrieval_notes.append(f"targeted_feedback_lookup:{topic}")

        # de-duplicate by source_hint/title
        deduped = []
        seen = set()
        for item in findings:
            key = (item.get("source_hint"), item.get("title"))
            if key not in seen:
                deduped.append(item)
                seen.add(key)

        return SearchResult(
            company=request.company,
            topic=request.topic,
            findings=deduped,
            retrieval_notes=retrieval_notes,
        )