import json
import socket
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from main import run_debate
from simulation.rag_agent import run_speech
from state import create_initial_state
from tools.mock_llm import MockRagModel, mock_search


class ScriptedModel(BaseChatModel):
    replies: list[Any]
    seen: list = Field(default_factory=list)
    @property
    def _llm_type(self):
        return 'scripted-test'
    def bind_tools(self, tools, **kwargs):
        return self
    def _generate(self, messages, **kwargs):
        self.seen.append(list(messages))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ChatResult(generations=[ChatGeneration(message=reply)])


def call(identifier, query='test', name='search_local_documents'):
    return {'name': name, 'args': {'query': query}, 'id': identifier, 'type': 'tool_call'}


def speak(model, search=mock_search, callback=None):
    return run_speech(model, search, create_initial_state('议题'), 'government', '陈述方案', 1, callback)


def test_agent_can_speak_without_search():
    result = speak(ScriptedModel(replies=[AIMessage(content='无需引用的立场陈述。')]),
                   search=lambda q: pytest.fail('unexpected search'))
    assert result['tool_logs'] == [] and result['records'][0]['evidence_ids'] == []


def test_budget_cache_and_tool_protocol():
    searches = []
    def search(query):
        searches.append(query)
        return mock_search(query)
    model = ScriptedModel(replies=[
        AIMessage(content='', tool_calls=[call('a'), call('b'), call('c')]),
        AIMessage(content='基于资料 [E:offline-evidence-1]')])
    result = speak(model, search)
    assert searches == ['test']
    assert [log['status'] for log in result['tool_logs']] == ['ok', 'ok', 'budget_exceeded']
    assert result['tool_logs'][1]['cached']
    messages = [m for m in model.seen[-1] if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in messages] == ['a', 'b', 'c']
    assert result['records'][0]['evidence_ids'] == ['offline-evidence-1']


@pytest.mark.parametrize('kind', ['unknown', 'empty', 'exception', 'invalid_argument'])
def test_tool_errors_do_not_crash_speech(kind):
    def search(query):
        if kind == 'exception':
            raise TimeoutError('test timeout')
        return []
    request = call('a', '' if kind == 'invalid_argument' else 'test',
                   'not_a_tool' if kind == 'unknown' else 'search_local_documents')
    result = speak(ScriptedModel(replies=[AIMessage(content='', tool_calls=[request]),
                                         AIMessage(content='证据不足。')]), search)
    assert result['warnings']
    assert result['records'][0]['content'] == '证据不足。'


@pytest.mark.parametrize('repaired', [True, False])
def test_bad_citation_one_repair(repaired):
    model = ScriptedModel(replies=[AIMessage(content='主张 [E:invented]'),
                                  AIMessage(content='缺少证据。' if repaired else '主张 [E:invented]')])
    result = speak(model)
    assert len(model.seen) == 2
    assert result['records'][0]['evidence_ids'] == []
    assert bool(result['warnings']) is not repaired
    assert result['records'][0]['invalid_evidence_ids'] == ([] if repaired else ['invented'])


def test_tool_results_persist_before_generation_failure(tmp_path):
    model = ScriptedModel(replies=[AIMessage(content='开场'),
        AIMessage(content='', tool_calls=[call('search')]), TimeoutError('final generation failed')])
    code, path = run_debate('议题', 1, tmp_path, rag=True, llm=model, search=mock_search)
    data = json.loads((path / 'result.json').read_text())
    assert code == 1 and len(data['records']) == 1
    assert len(data['tool_logs']) == 1 and 'offline-evidence-1' in data['evidence']


def test_offline_rag_no_network_and_no_duplicates(tmp_path, monkeypatch):
    monkeypatch.setattr(socket.socket, 'connect', lambda *args: pytest.fail('network used'))
    code, path = run_debate('议题', 3, tmp_path, mock=True, rag=True)
    data = json.loads((path / 'result.json').read_text())
    assert code == 0 and len(data['records']) == 10
    assert len(data['tool_logs']) == 6
    assert len({log['id'] for log in data['tool_logs']}) == 6
    assert data['evaluation']['evidence_comment']
    assert all(r['evidence_ids'] for r in data['records'] if r['role'] != 'host')
    _, other = run_debate('另一场', 1, tmp_path, mock=True, rag=True)
    assert len(json.loads((other / 'result.json').read_text())['tool_logs']) == 2


def test_judge_receives_evidence_and_full_transcript():
    from evaluation.evaluator import evaluate
    from tools.mock_llm import mock_review
    review = mock_review()
    review['evidence_comment'] = '研究范围有限。'
    model = ScriptedModel(replies=[AIMessage(content=json.dumps(review))])
    records = [{'role': r, 'round': 1, 'content': f'{r} [E:offline-evidence-1]'} for r in ['government', 'opposition']]
    evaluate(model, '议题', records, evidence={'offline-evidence-1': mock_search('')[0]})
    prompt = '\n'.join(m.content for m in model.seen[0])
    assert all(r['content'] in prompt for r in records)
    assert 'FICTIONAL TEST FIXTURE' in prompt
