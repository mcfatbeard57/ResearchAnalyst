from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from src.schemas import ConversationTurn


class GraphState(TypedDict, total=False):
    # current turn input
    user_query: str
    resolved_query: Optional[str]

    # memory
    conversation_history: List[dict]
    active_company: Optional[str]
    last_topic: Optional[str]

    # clarity stage
    clarity_status: str
    clarification_question: Optional[str]

    # research stage
    research_output: Optional[dict]
    rule_confidence_score: Optional[float]
    llm_confidence_score: Optional[float]
    confidence_gate_passed: Optional[bool]

    # validation stage
    validation_result: Optional[str]
    validation_feedback: Optional[str]

    # retry control
    attempt_count: int

    # final stage
    final_response: Optional[str]


def make_initial_state(
    user_query: str,
    conversation_history: Optional[List[dict]] = None,
    active_company: Optional[str] = None,
    last_topic: Optional[str] = None,
) -> GraphState:
    return GraphState(
        user_query=user_query,
        resolved_query=None,
        conversation_history=conversation_history or [],
        active_company=active_company,
        last_topic=last_topic,
        clarity_status="needs_clarification",
        clarification_question=None,
        research_output=None,
        rule_confidence_score=None,
        llm_confidence_score=None,
        confidence_gate_passed=None,
        validation_result=None,
        validation_feedback=None,
        attempt_count=0,
        final_response=None,
    )