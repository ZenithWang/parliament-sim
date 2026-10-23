"""Only new records are returned by nodes; the reducer appends them once."""
from operator import add
from typing import Annotated, Literal, TypedDict, NotRequired


class Record(TypedDict):
    role: Literal["host", "government", "opposition"]
    round: int
    content: str
    evidence_ids: NotRequired[list[str]]
    invalid_evidence_ids: NotRequired[list[str]]


def merge_evidence(old, new):
    return {**old, **new}


class ParliamentState(TypedDict):
    topic: str
    rounds: int
    completed_rounds: int
    records: Annotated[list[Record], add]
    evaluation: dict | None
    evidence: Annotated[dict, merge_evidence]
    tool_logs: Annotated[list[dict], add]
    warnings: Annotated[list[str], add]


def create_initial_state(topic: str, rounds: int = 3) -> ParliamentState:
    if not topic.strip():
        raise ValueError("议题不能为空。")
    if isinstance(rounds, bool) or not isinstance(rounds, int) or not 1 <= rounds <= 5:
        raise ValueError("辩论轮数必须为 1–5。")
    return {"topic": topic.strip(), "rounds": rounds, "completed_rounds": 0,
            "records": [], "evaluation": None, "evidence": {}, "tool_logs": [], "warnings": []}
