from typing import Final, Literal, get_args

TOPIC_OVERVIEW: Final[str] = "overview"
TOPIC_NEWS: Final[str] = "news"
TOPIC_FINANCIALS: Final[str] = "financials"
TOPIC_CEO: Final[str] = "ceo"
TOPIC_COMPETITORS: Final[str] = "competitors"
TOPIC_DEVELOPMENTS: Final[str] = "developments"

SUPPORTED_TOPICS = [
    TOPIC_OVERVIEW,
    TOPIC_NEWS,
    TOPIC_FINANCIALS,
    TOPIC_CEO,
    TOPIC_COMPETITORS,
    TOPIC_DEVELOPMENTS,
]

ClarityStatus = Literal["clear", "needs_clarification"]
ValidationResult = Literal["sufficient", "insufficient"]

MAX_RESEARCH_ATTEMPTS: Final[int] = 3
CONFIDENCE_THRESHOLD: Final[float] = 6.0