from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

from src.constants import (
    TOPIC_OVERVIEW,
    TOPIC_NEWS,
    TOPIC_FINANCIALS,
    TOPIC_CEO,
    TOPIC_COMPETITORS,
    TOPIC_DEVELOPMENTS,
)
from src.schemas import ClarityOutput


TOPIC_KEYWORDS: Dict[str, List[str]] = {
    TOPIC_NEWS: [
        "news", "headline", "headlines", "recent news", "latest news", "updates"
    ],
    TOPIC_FINANCIALS: [
        "financial", "financials", "stock", "earnings", "revenue", "profit",
        "market cap", "income", "quarterly", "results"
    ],
    TOPIC_CEO: [
        "ceo", "chief executive", "leadership", "leader", "founder"
    ],
    TOPIC_COMPETITORS: [
        "competitor", "competitors", "competition", "rivals", "rival"
    ],
    TOPIC_DEVELOPMENTS: [
        "developments", "development", "launch", "launches", "roadmap",
        "product", "products", "acquisition", "acquisitions", "recent developments"
    ],
    TOPIC_OVERVIEW: [
        "overview", "about", "tell me about", "what is", "who are they", "how are they doing"
    ],
}

NON_COMPANY_PHRASES = {
    "the ceo",
    "ceo",
    "competitors",
    "competitor",
    "competition",
    "rivals",
    "recent news",
    "latest news",
    "news",
    "financials",
    "financial",
    "stock",
    "their stock",
    "developments",
    "recent developments",
    "that company",
    "this company",
    "the company",
}

FOLLOW_UP_ONLY_PATTERNS = [
    r"^what about\??$",
    r"^what about .+",
    r"^tell me more\??$",
    r"^more\??$",
    r"^and\??$",
    r"^what next\??$",
]

COMPARE_PATTERNS = [
    r"\bcompare\b",
    r"\bvs\b",
    r"\bversus\b",
]

REFERENTIAL_PATTERNS = [
    r"\bthat company\b",
    r"\bthis company\b",
    r"\bthe company\b",
    r"\bthey\b",
    r"\btheir\b",
    r"\bthem\b",
    r"\bit\b",
]


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _canonicalize_company_name(name: str) -> str:
    name = _normalize_spaces(name.strip(" ?!.,:;"))
    if not name:
        return name

    tokens = []
    for token in name.split():
        if token.isupper():
            tokens.append(token)
        elif token.lower() in {"inc", "corp", "ltd", "llc", "plc"}:
            tokens.append(token.upper() if len(token) <= 3 else token.title())
        elif token.lower() in {"and", "&", "of", "the"}:
            tokens.append(token.lower() if token.lower() != "the" else "The")
        else:
            tokens.append(token.title())
    return " ".join(tokens)


def _detect_topic(query: str, last_topic: Optional[str] = None) -> str:
    q = query.lower()

    scores = {
        TOPIC_OVERVIEW: 0,
        TOPIC_NEWS: 0,
        TOPIC_FINANCIALS: 0,
        TOPIC_CEO: 0,
        TOPIC_COMPETITORS: 0,
        TOPIC_DEVELOPMENTS: 0,
    }

    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            if keyword in q:
                scores[topic] += len(keyword.split())

    best_topic = max(scores, key=scores.get)
    best_score = scores[best_topic]

    if best_score == 0:
        if last_topic and _is_follow_up_only_query(query):
            return last_topic
        return TOPIC_OVERVIEW

    return best_topic


def _is_follow_up_only_query(query: str) -> bool:
    q = _normalize_spaces(query.lower())
    return any(re.match(pattern, q) for pattern in FOLLOW_UP_ONLY_PATTERNS)


def _has_referential_language(query: str) -> bool:
    q = query.lower()
    return any(re.search(pattern, q) for pattern in REFERENTIAL_PATTERNS)


def _looks_like_multi_company_request(query: str) -> bool:
    q = query.lower()
    if any(re.search(pattern, q) for pattern in COMPARE_PATTERNS):
        return True

    # Very narrow V1 rule: "compare Apple and Tesla" style
    if q.startswith("compare ") and " and " in q:
        return True

    return False


def _clean_candidate(candidate: str) -> str:
    candidate = _normalize_spaces(candidate.strip(" ?!.,:;"))
    candidate = re.sub(r"^(the|a|an)\s+", "", candidate, flags=re.IGNORECASE)

    trailing_noise = [
        " competitors", " competitor", " competition", " rivals", " rival",
        " news", " latest news", " recent news",
        " financials", " financial", " stock",
        " ceo", " leadership", " developments", " recent developments",
    ]

    lowered = candidate.lower()
    for suffix in sorted(trailing_noise, key=len, reverse=True):
        if lowered.endswith(suffix):
            candidate = candidate[: -len(suffix)].strip()
            lowered = candidate.lower()

    return candidate


def _is_valid_company_candidate(candidate: str) -> bool:
    if not candidate:
        return False

    lowered = candidate.lower().strip()
    if lowered in NON_COMPANY_PHRASES:
        return False

    # Too generic
    if lowered in {"him", "her", "them", "they", "it"}:
        return False

    # Avoid very long free-form spans
    if len(candidate.split()) > 6:
        return False

    return True


def _extract_companies_from_catalog(query: str, known_companies: Optional[Sequence[str]]) -> List[str]:
    if not known_companies:
        return []

    q = query.lower()
    matches = []

    for company in sorted(set(known_companies), key=len, reverse=True):
        if company.lower() in q:
            matches.append(_canonicalize_company_name(company))

    # de-dupe preserve order
    deduped = []
    seen = set()
    for item in matches:
        if item.lower() not in seen:
            deduped.append(item)
            seen.add(item.lower())
    return deduped


def _extract_company_by_pattern(query: str) -> List[str]:
    q = _normalize_spaces(query)

    patterns = [
        r"\b(?:about|on|for)\s+(?P<company>[A-Za-z0-9&.\- ]+?)(?:\?|$)",
        r"\b(?:ceo|chief executive|financials|competitors|news|developments)\s+(?:of|for)\s+(?P<company>[A-Za-z0-9&.\- ]+?)(?:\?|$)",
        r"^(?P<company>[A-Za-z0-9&.\- ]+?)\s+(?:news|financials|competitors|stock|ceo|developments)\b",
    ]

    candidates = []
    for pattern in patterns:
        match = re.search(pattern, q, flags=re.IGNORECASE)
        if match:
            candidate = _clean_candidate(match.group("company"))
            if _is_valid_company_candidate(candidate):
                candidates.append(_canonicalize_company_name(candidate))

    # de-dupe preserve order
    deduped = []
    seen = set()
    for item in candidates:
        if item.lower() not in seen:
            deduped.append(item)
            seen.add(item.lower())
    return deduped


def _extract_explicit_companies(query: str, known_companies: Optional[Sequence[str]] = None) -> List[str]:
    catalog_matches = _extract_companies_from_catalog(query, known_companies)
    pattern_matches = _extract_company_by_pattern(query)
    combined = catalog_matches + pattern_matches

    deduped = []
    seen = set()
    for item in combined:
        if item.lower() not in seen:
            deduped.append(item)
            seen.add(item.lower())
    return deduped


def _resolve_company_from_history(
    conversation_history: Optional[List[dict]],
    known_companies: Optional[Sequence[str]] = None,
) -> Optional[str]:
    if not conversation_history:
        return None

    # search most recent user turns first
    for turn in reversed(conversation_history[-6:]):
        message = turn.get("message", "")
        role = turn.get("role", "")
        if role != "user":
            continue

        companies = _extract_explicit_companies(message, known_companies=known_companies)
        if len(companies) == 1:
            return companies[0]

    return None


def _build_resolved_query(company: str, topic: str) -> str:
    if topic == TOPIC_OVERVIEW:
        return f"Tell me about {company}"
    if topic == TOPIC_NEWS:
        return f"Tell me about recent news for {company}"
    if topic == TOPIC_FINANCIALS:
        return f"Tell me about {company} financials"
    if topic == TOPIC_CEO:
        return f"Tell me about the CEO of {company}"
    if topic == TOPIC_COMPETITORS:
        return f"Tell me about {company} competitors"
    if topic == TOPIC_DEVELOPMENTS:
        return f"Tell me about recent developments at {company}"
    return f"Tell me about {company}"


class ClarityAgent:
    def __init__(self, known_companies: Optional[Sequence[str]] = None):
        self.known_companies = list(known_companies) if known_companies else None

    def run(self, state: dict) -> ClarityOutput:
        user_query = _normalize_spaces(state["user_query"])
        active_company = state.get("active_company")
        last_topic = state.get("last_topic")
        conversation_history = state.get("conversation_history", [])

        if _looks_like_multi_company_request(user_query):
            return ClarityOutput(
                clarity_status="needs_clarification",
                active_company=None,
                last_topic=None,
                resolved_query=None,
                clarification_question="I can handle one company at a time in this version. Which company should I research?",
            )

        explicit_companies = _extract_explicit_companies(
            user_query,
            known_companies=self.known_companies,
        )

        if len(explicit_companies) > 1:
            return ClarityOutput(
                clarity_status="needs_clarification",
                active_company=None,
                last_topic=None,
                resolved_query=None,
                clarification_question="I found multiple companies in your request. Please name one company for this version.",
            )

        detected_topic = _detect_topic(user_query, last_topic=last_topic)

        if len(explicit_companies) == 1:
            company = explicit_companies[0]
            return ClarityOutput(
                clarity_status="clear",
                active_company=company,
                last_topic=detected_topic,
                resolved_query=_build_resolved_query(company, detected_topic),
                clarification_question=None,
            )

        history_company = _resolve_company_from_history(
            conversation_history,
            known_companies=self.known_companies,
        )
        resolved_company = active_company or history_company

        if resolved_company:
            if detected_topic:
                return ClarityOutput(
                    clarity_status="clear",
                    active_company=resolved_company,
                    last_topic=detected_topic,
                    resolved_query=_build_resolved_query(resolved_company, detected_topic),
                    clarification_question=None,
                )

            if _is_follow_up_only_query(user_query) and last_topic:
                return ClarityOutput(
                    clarity_status="clear",
                    active_company=resolved_company,
                    last_topic=last_topic,
                    resolved_query=_build_resolved_query(resolved_company, last_topic),
                    clarification_question=None,
                )

            if _has_referential_language(user_query):
                fallback_topic = last_topic or TOPIC_OVERVIEW
                return ClarityOutput(
                    clarity_status="clear",
                    active_company=resolved_company,
                    last_topic=fallback_topic,
                    resolved_query=_build_resolved_query(resolved_company, fallback_topic),
                    clarification_question=None,
                )

        # no deterministic company resolution possible
        if _has_referential_language(user_query) or _is_follow_up_only_query(user_query):
            return ClarityOutput(
                clarity_status="needs_clarification",
                active_company=None,
                last_topic=None,
                resolved_query=None,
                clarification_question="Which company are you asking about?",
            )

        return ClarityOutput(
            clarity_status="needs_clarification",
            active_company=None,
            last_topic=None,
            resolved_query=None,
            clarification_question=(
                "Please tell me the company name. You can also specify a topic like overview, "
                "news, financials, CEO, competitors, or developments."
            ),
        )