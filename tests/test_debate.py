import json
import socket

import pytest
from langchain_core.runnables import RunnableLambda

from evaluation.evaluator import EvaluationError, evaluate
from main import main, run_debate
from state import create_initial_state
from tools.mock_llm import MockLLM, mock_review, respond
from workflow import build_parliament_graph


@pytest.mark.parametrize('rounds', [1, 3, 5])
def test_order_context_and_isolation(rounds):
    prompts = []
    def model(prompt):
        prompts.append(prompt.to_string())
        return respond(prompt)
    app = build_parliament_graph(RunnableLambda(model))
    initial = create_initial_state('测试议题', rounds)
    result = app.invoke(initial, {'recursion_limit': 3 * rounds + 10})
    records = result['records']
    assert [r['role'] for r in records] == ['host'] + ['government', 'opposition', 'host'] * rounds
    assert [r['round'] for r in records] == [0] + [r for r in range(1, rounds + 1) for _ in range(3)]
    assert result['completed_rounds'] == rounds
    assert result['evaluation']['winner'] == 'tie'
    assert len(prompts) == len(records) + 1
    for i, prompt in enumerate(prompts):
        for record in records[:i]:
            assert record['content'] in prompt
    assert initial['records'] == []
    again = app.invoke(create_initial_state('另一个议题', 1))
    assert len(again['records']) == 4


@pytest.mark.parametrize('bad', ['json', 'range', 'reference'])
def test_judge_repairs_once(bad):
    records = [{'role': r, 'round': 1, 'content': r} for r in ('government', 'opposition')]
    review = mock_review()
    if bad == 'range':
        review['government']['coherence']['score'] = 11
    if bad == 'reference':
        review['government']['coherence']['references'][0]['round'] = 5
    invalid = 'not json' if bad == 'json' else json.dumps(review)
    prompts = []
    def model(prompt):
        prompts.append(prompt.to_string())
        return invalid if len(prompts) == 1 else json.dumps(mock_review())
    assert evaluate(RunnableLambda(model), '议题', records)['winner'] == 'tie'
    assert len(prompts) == 2
    assert 'government' in prompts[1] and '上次输出校验失败' in prompts[1]


def test_failed_judge_keeps_all_records(tmp_path):
    calls = []
    def model(prompt):
        if '独立评审' in prompt.to_messages()[0].content:
            calls.append(1)
            return 'invalid'
        return respond(prompt)
    code, directory = run_debate('议题', 3, tmp_path, llm=RunnableLambda(model))
    data = json.loads((directory / 'result.json').read_text())
    assert code == 1 and len(calls) == 2
    assert len(data['records']) == 10
    assert data['evaluation'] is None and data['status'] == 'failed'
    assert 'EvaluationError' in data['error']


def test_timeout_saves_partial_and_each_node(tmp_path):
    calls = []
    def model(prompt):
        calls.append(1)
        if len(calls) == 2:
            data = json.loads(next(tmp_path.glob('*/result.json')).read_text())
            assert len(data['records']) == 1  # persisted before next invocation
            raise TimeoutError('模型请求超时')
        return respond(prompt)
    code, directory = run_debate('议题', output=tmp_path, llm=RunnableLambda(model))
    data = json.loads((directory / 'result.json').read_text())
    assert code == 1 and len(data['records']) == 1
    assert 'TimeoutError' in data['error']


def test_offline_no_network_or_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    monkeypatch.setattr(socket.socket, 'connect', lambda *a: pytest.fail('network used'))
    code, directory = run_debate('议题', output=tmp_path, mock=True)
    data = json.loads((directory / 'result.json').read_text())
    assert code == 0 and data['status'] == 'completed'
    assert len(data['records']) == 10
    assert '离线模拟' in (directory / 'transcript.md').read_text()
    _, other = run_debate('第二场', 1, tmp_path, mock=True)
    assert other != directory


def test_missing_key(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY', '')
    code, directory = run_debate('议题', output=tmp_path)
    assert code == 1
    data = json.loads((directory / 'result.json').read_text())
    assert 'DEEPSEEK_API_KEY' in data['error'] and data['records'] == []


@pytest.mark.parametrize('args', [['--topic', ' '], ['--rounds', '0'], ['--rounds', '6'], ['--strategy', 'baseline']])
def test_cli_rejects_invalid_input(args):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2


def test_provider_timeout_and_retry(monkeypatch):
    import langchain_deepseek
    from config import create_llm
    captured = {}
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-secret')
    monkeypatch.setenv('DEEPSEEK_MODEL', 'deepseek-chat')
    monkeypatch.setattr(langchain_deepseek, 'ChatDeepSeek', lambda **kw: captured.update(kw))
    create_llm()
    assert captured['timeout'] == 60 and captured['max_retries'] == 1


def test_secret_redacted():
    import os
    from unittest.mock import patch
    from main import safe_error
    with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'secret-value'}):
        assert 'secret-value' not in safe_error(ValueError('failed secret-value'))
