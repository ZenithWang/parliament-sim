"""Fixed debate routing with optional per-speech RAG agents; no checkpoints."""
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph

from evaluation.evaluator import evaluate
from simulation.prompts import history, speech_prompt
from state import ParliamentState


def build_parliament_graph(llm, on_start=None, search=None, on_tool_event=None):
    chains = {role: speech_prompt(role) | llm | StrOutputParser()
              for role in ("host", "government", "opposition")}

    def speak(state, role, task, round_number):
        if on_start:
            on_start(role, round_number)
        if search is not None and role != 'host':
            from simulation.rag_agent import run_speech
            return run_speech(llm, search, state, role, task, round_number, on_tool_event)
        content = chains[role].invoke({**state, "round": round_number,
                                      "task": task, "history": history(state["records"])}).strip()
        if not content:
            raise ValueError("模型返回空发言。")
        return {"records": [{"role": role, "round": round_number, "content": content}]}

    def opening(state):
        return speak(state, "host", "简短介绍议题、双方立场和辩论规则。", 0)

    def government(state):
        return speak(state, "government", "约150–250个中文字；首次陈述方案，后续明确回应对方上一轮质疑。",
                     state["completed_rounds"] + 1)

    def opposition(state):
        return speak(state, "opposition", "约150–250个中文字，直接回应当轮政府方主张。",
                     state["completed_rounds"] + 1)

    def summary(state):
        round_number = state["completed_rounds"] + 1
        return {**speak(state, "host", "简短总结本轮主要争议，保持中立，不评判胜负。", round_number),
                "completed_rounds": round_number}

    def judge(state):
        if on_start:
            on_start("judge", state["completed_rounds"])
        return {"evaluation": evaluate(llm, state["topic"], state["records"],
                evidence=state.get("evidence", {}) if search is not None else None,
                warnings=state.get("warnings", []))}

    graph = StateGraph(ParliamentState)
    for name, node in (("opening", opening), ("government", government),
                       ("opposition", opposition), ("summary", summary), ("judge", judge)):
        graph.add_node(name, node)
    graph.add_edge(START, "opening")
    graph.add_edge("opening", "government")
    graph.add_edge("government", "opposition")
    graph.add_edge("opposition", "summary")
    graph.add_conditional_edges("summary", lambda s: "judge" if s["completed_rounds"] >= s["rounds"] else "government",
                                {"judge": "judge", "government": "government"})
    graph.add_edge("judge", END)
    return graph.compile()
