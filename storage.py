"""Save after each graph update. Each file is replaced atomically."""
import json
from pathlib import Path
from simulation.roles import LABELS
from evaluation.evaluator import DIMENSIONS


def render_markdown(data):
    lines = ["# 议会辩论记录", f"议题：{data['topic']}",
             f"模式：{'离线模拟（固定示例，不是真实模型评分）' if data['config']['mock'] else 'DeepSeek 模型辩论'}",
             f"状态：{data['status']}；耗时：{data['duration_s']:.2f} 秒",
             f"配置：`{json.dumps(data['config'], ensure_ascii=False)}`"]
    for record in data['records']:
        lines += [f"## 第{record['round']}轮 · {LABELS[record['role']]}", record['content']]
    review = data.get('evaluation')
    if review:
        lines += ["## 评审（模拟判断，不是客观事实标签）", f"胜方：{LABELS[review['winner']]}",
                  review['winner_reason'], review['summary']]
        for role in ('government', 'opposition'):
            lines.append(f"### {LABELS[role]}")
            for dimension in DIMENSIONS:
                score = review[role][dimension]
                refs = '、'.join(f"{LABELS[r['role']]}第{r['round']}轮" for r in score['references'])
                lines.append(f"- {dimension}：{score['score']}/10。{score['reason']}（依据：{refs}）")
            lines += [f"优点：{review[role]['strengths']}", f"不足：{review[role]['weaknesses']}"]
    if review and review.get('evidence_comment'):
        lines += ['### 证据使用点评', review['evidence_comment']]
    if data.get('tool_logs'):
        lines.append('## 检索过程')
        for log in data['tool_logs']:
            lines.append(f"- {LABELS[log['role']]} 第{log['round']}轮：{log['query']}；状态 {log['status']}；缓存 {log['cached']}；耗时 {log['duration_s']:.2f}s；片段 {', '.join(log['evidence_ids']) or '无'}")
    if data.get('evidence'):
        lines.append('## 引用资料（检索候选，不等于事实核验）')
        for identifier, item in data['evidence'].items():
            location = f"PDF物理页码 {item['page']}" if item.get('page') else item.get('heading', '')
            lines += [f"### [E:{identifier}] {item['title']}",
                      f"文件：{item['filename']}；{location}；日期：{item.get('publication_date', '')}",
                      f"来源：{item.get('url', '') or '本地文件'}", item['content']]
    if data.get('warnings'):
        lines += ['## 警告', *data['warnings']]
    if data.get('error'):
        lines += ["## 失败信息", data['error']]
    return '\n\n'.join(lines) + '\n'


def save_run(directory: Path, data):
    for filename, text in (("result.json", json.dumps(data, ensure_ascii=False, indent=2)),
                           ("transcript.md", render_markdown(data))):
        path = directory / filename
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(text, encoding='utf-8')
        temporary.replace(path)
