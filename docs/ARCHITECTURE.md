# 架构与数据流

## 状态

`ParliamentState` 是 TypedDict，包含 `topic`、`rounds`、`completed_rounds`、`records`、`evaluation`。记录使用 `Annotated[list[Record], operator.add]` 追加 reducer。每条记录包含 `role`、`round` 和 `content`；开场为第 0 轮。主持总结节点递增已完成轮数，是唯一更新轮数的位置。

每次调用 `create_initial_state` 返回独立空列表。图没有 checkpointer 和跨会话记忆。图构建函数 `build_parliament_graph(llm, on_start=None)` 接受 LangChain Runnable 模型，真实模型与假模型使用相同节点和路由。

## 调用链

普通角色链：提示词 → 模型 → 文本解析器。上下文是带角色和轮次标记的完整记录，放在 human 消息中；system 消息固定角色职责与规则。第一版最多五轮，不做摘要压缩。

评审链：提示词（包括 Pydantic JSON schema）→ 模型 → 文本 → Pydantic 解析与引用校验。每一评分项引用必须匹配已存在的 `(role, round)`。校验不能验证理由是否准确解释原文，真实演示需要人工阅读。

修正链保留原评审提示词及完整历史，附加上次错误输出与校验错误。网络异常不会伪装成格式异常；SDK负责请求重试，评审逻辑仅处理一次格式修正。

## 执行和持久化

CLI 使用 `stream_mode='updates'`，逐个消费节点增量，不先收集整个 stream。对 records 做追加，对轮数和评审做覆盖，然后立即保存。CLI 的结果快照不传回图，避免重复追加。

每次运行一个独立目录。JSON 与 Markdown 分别原子替换，二者不是跨文件事务。状态为 running/completed/failed；耗时使用 monotonic 时钟。异常发生时保存最后一个已完成节点；未完成调用不写入记录，不提供恢复能力。

## 已移除功能

策略算法、算法对比、启发式评分、投票、Wikipedia、人工中断和 SQLite 检查点均已从活动代码移除。历史版本可通过 Git 或重构前备份查看，不维持旧 CLI 兼容性。


## 第二版补充

以上描述第一版基础流程。第二版通过显式 `--rag` 为政府和反对方增加独立工具循环，共享状态新增 evidence、tool_logs 和 warnings；工具节点完成即通过回调保存，外层增量去重。Redis只保存知识库向量，不作为辩论检查点。详见 [第二版讲解](V2_WALKTHROUGH.md)。
