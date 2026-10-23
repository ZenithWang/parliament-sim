"""Deterministic OFFLINE fixture, not an actual model quality evaluation."""
import json
import re
from langchain_core.runnables import RunnableLambda


def mock_review():
    result = {"winner": "tie", "winner_reason": "离线示例：双方论证各有优点，不代表真实模型判断。",
              "summary": "离线演示：政策收益与实施风险需要进一步验证。"}
    for role in ("government", "opposition"):
        result[role] = {dimension: {"score": 7, "reason": "离线示例：第一轮提出了明确主张。",
                                   "references": [{"role": role, "round": 1}]}
                        for dimension in ("coherence", "responsiveness", "feasibility", "persuasiveness")}
        result[role].update(strengths="提出了明确观点。", weaknesses="仍需可靠证据支撑。")
    return result


def respond(prompt):
    messages = prompt.to_messages()
    system, user = messages[0].content, messages[-1].content
    if "独立评审" in system:
        return json.dumps(mock_review(), ensure_ascii=False)
    round_number = re.search(r"当前轮次：(\d+)", user).group(1)
    if "你是政府方" in system:
        text = "我们支持议题主张，建议分阶段试点，以公开指标评估收益和成本，并回应对方对执行风险的担忧。"
    elif "你是反对党" in system:
        text = "政府提出分阶段试点，但尚未说明成本如何分担。我们建议明确适用范围、退出机制与独立评估条件。"
    else:
        text = "双方请围绕议题有序发言。本次讨论关注政策收益、实施条件与风险，请回应对方论点。"
    return f"【离线模拟，第{round_number}轮】{text}"


def MockLLM():
    return RunnableLambda(respond)


# Tool-capable offline model. It emits actual AIMessage tool_calls and consumes
# ToolMessage responses; the application runs exactly the real agent subgraph.
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompt_values import ChatPromptValue


class MockRagModel(BaseChatModel):
    @property
    def _llm_type(self):
        return 'offline-rag-fixture'

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        system = messages[0].content
        if '独立评审' in system:
            review = mock_review()
            if 'evidence_comment' in system:
                review['evidence_comment'] = '离线示例：引用来自固定测试片段，只验证引用链路，不代表真实研究结论。'
            answer = AIMessage(content=json.dumps(review, ensure_ascii=False))
        elif 'search_local_documents' in messages[1].content:
            tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
            if not tool_messages:
                answer = AIMessage(content='', tool_calls=[{'name': 'search_local_documents',
                    'args': {'query': '四天工作制的实施困难与研究局限'}, 'id': 'offline-search', 'type': 'tool_call'}])
            else:
                payload = json.loads(tool_messages[-1].content)
                items = payload.get('evidence', [])
                citation = f" [E:{items[0]['chunk_id']}]" if items else ' 没有足够证据。'
                answer = AIMessage(content='【离线模拟】试点效果应结合样本范围和实施条件理解，不能推广到所有行业。' + citation)
        else:
            answer = AIMessage(content=respond(ChatPromptValue(messages=messages)))
        return ChatResult(generations=[ChatGeneration(message=answer)])


def mock_search(query):
    return [{'chunk_id': 'offline-evidence-1', 'content': 'FICTIONAL TEST FIXTURE: outcomes depend on implementation and sample selection.',
             'title': '离线虚构资料', 'filename': 'offline-fixture.md', 'page': 0,
             'heading': '示例', 'url': '', 'publication_date': '', 'publisher': '测试', 'file_hash': 'mock'}]
