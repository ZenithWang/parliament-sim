"""CLI entry point; streams and persists each completed node immediately."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from time import monotonic
from uuid import uuid4

from config import DEFAULT_TOPIC, create_llm, load_config
from simulation.roles import LABELS
from state import create_initial_state
from storage import render_markdown, save_run
from workflow import build_parliament_graph


from security import safe_error


def run_debate(topic, rounds=3, output='results', mock=False, llm=None, rag=False, search=None):
    state = create_initial_state(topic, rounds)
    settings = load_config()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid4().hex[:8]
    directory = Path(output).resolve() / run_id
    directory.mkdir(parents=True, exist_ok=False)
    start = monotonic()
    data = {**state, 'records': [], 'evidence': {}, 'tool_logs': [], 'warnings': [], 'run_id': run_id, 'status': 'running', 'duration_s': 0,
            'config': {**settings, 'mock': mock, 'rounds': rounds, 'rag': rag}, 'error': None}

    def persist():
        data['duration_s'] = monotonic() - start
        save_run(directory, data)

    def apply_tools(update):
        data['evidence'].update(update.get('evidence', {}))
        existing = {log['id'] for log in data['tool_logs']}
        data['tool_logs'].extend(log for log in update.get('tool_logs', []) if log['id'] not in existing)
        for warning in update.get('warnings', []):
            if warning not in data['warnings']:
                data['warnings'].append(warning)

    def on_tool_event(update):
        apply_tools(update)
        persist()
        for log in update.get('tool_logs', []):
            print(f"检索 [{log['status']}] {log['query']} → {len(log['evidence_ids'])} 个片段", flush=True)

    def on_start(role, round_number):
        print(f"\n正在生成：{LABELS[role]} · 第{round_number}轮……", flush=True)

    print(f"结果目录：{directory}", flush=True)
    if mock:
        print('【离线模拟】固定示例内容，不调用模型，不代表真实评分。', flush=True)
    persist()
    try:
        if rag:
            if mock:
                from tools.mock_llm import MockRagModel, mock_search
                model = llm if llm is not None else MockRagModel()
                search = search if search is not None else mock_search
            else:
                model = llm if llm is not None else create_llm()
                if search is None:
                    from knowledge.store import KnowledgeStore
                    search = KnowledgeStore.open_for_search().search
        else:
            model = llm if llm is not None else create_llm(mock)
            search = None
        app = build_parliament_graph(model, on_start=on_start, search=search, on_tool_event=on_tool_event)
        for event in app.stream(state, {'recursion_limit': 3 * rounds + 10}, stream_mode='updates'):
            for update in event.values():
                apply_tools(update)
                if 'records' in update:
                    data['records'].extend(update['records'])
                for key in ('completed_rounds', 'evaluation'):
                    if key in update:
                        data[key] = update[key]
                persist()
                for record in update.get('records', []):
                    print(record['content'], flush=True)
        data['status'] = 'completed'
    except (Exception, KeyboardInterrupt) as error:
        data['status'] = 'failed'
        data['error'] = safe_error(error)
        print(f"运行失败：{data['error']}", file=sys.stderr, flush=True)
    persist()
    if data['evaluation']:
        print('\n## 评审' + render_markdown(data).split('## 评审', 1)[1], flush=True)
    print(f"已保存：{directory / 'result.json'}\n已保存：{directory / 'transcript.md'}", flush=True)
    return (0 if data['status'] == 'completed' else 1), directory


def main(argv=None):
    parser = argparse.ArgumentParser(description='LangChain + LangGraph 中文多智能体议会辩论')
    parser.add_argument('--topic', default=DEFAULT_TOPIC)
    parser.add_argument('--rounds', type=int, choices=range(1, 6), default=3)
    parser.add_argument('--output', default='results', help='结果根目录，每次运行创建独立子目录')
    parser.add_argument('--rag', action='store_true', help='启用本地知识库检索工具')
    parser.add_argument('--mock-llm', action='store_true', help='离线固定示例，无需密钥或网络')
    args = parser.parse_args(argv)
    if not args.topic.strip():
        parser.error('议题不能为空。')
    try:
        code, _ = run_debate(args.topic, args.rounds, args.output, args.mock_llm, rag=args.rag)
        return code
    except OSError as error:
        print(f"无法保存结果：{safe_error(error)}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
