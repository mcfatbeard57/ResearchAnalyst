from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator

from src.constants import (
    SUPPORTED_TOPICS,
    CONFIDENCE_THRESHOLD,
)


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    message: str


class ClarityOutput(BaseModel):
    clarity_status: Literal["clear", "needs_clarification"]
    active_company: Optional[str] = None
    last_topic: Optional[str] = None
    resolved_query: Optional[str] = None
    clarification_question: Optional[str] = None

    @field_validator("last_topic")
    @classmethod
    def validate_topic(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if value not in SUPPORTED_TOPICS:
            raise ValueError(f"Invalid topic: {value}")
        return value


class ResearchItem(BaseModel):
    category: str = Field(..., description="news, financials, ceo, competitors, developments, etc.")
    title: str
    summary: str
    source_hint: Optional[str] = None


class ResearchOutput(BaseModel):
    company: str
    topic: str
    resolved_query: str
    findings: List[ResearchItem] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)

    rule_confidence_score: float = Field(..., ge=0, le=10)
    llm_confidence_score: float = Field(..., ge=0, le=10)
    confidence_gate_passed: bool

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, value: str) -> str:
        if value not in SUPPORTED_TOPICS:
            raise ValueError(f"Invalid topic: {value}")
        return value

    @field_validator("confidence_gate_passed")
    @classmethod
    def validate_gate_consistency(cls, value: bool, info):
        data = info.data
        rule_score = data.get("rule_confidence_score")
        llm_score = data.get("llm_confidence_score")

        if rule_score is not None and llm_score is not None:
            expected = (
                rule_score >= CONFIDENCE_THRESHOLD
                and llm_score >= CONFIDENCE_THRESHOLD
            )
            if value != expected:
                raise ValueError(
                    "confidence_gate_passed is inconsistent with confidence scores"
                )
        return value


class ValidationOutput(BaseModel):
    validation_result: Literal["sufficient", "insufficient"]
    validation_feedback: str


class SynthesisOutput(BaseModel):
    final_response: str