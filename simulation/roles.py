"""Four logical agents share one model, with distinct responsibilities."""
ROLES = {
    "host": "你是中立的议会主持人，维持秩序，不替任何一方辩论。",
    "government": "你是政府方，支持议题主张，提出可行方案、实施条件并回应反对意见。",
    "opposition": "你是反对党，挑战政府方案的假设、成本和公平性，并提出替代意见。",
    "judge": "你是独立评审，依据完整辩论上下文评价双方论证，公平说明理由。",
}
LABELS = {"host": "主持人", "government": "政府方", "opposition": "反对党", "judge": "评审", "tie": "平局"}
