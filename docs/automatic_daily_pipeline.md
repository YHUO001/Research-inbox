# 统一自动科研收件箱

## 调度（Asia/Singapore）

- `08:17`：主运行。Google Scholar 邮件与 OpenAlex 主动发现均执行。
- `20:47`：恢复运行。只执行当天尚未成功的来源；两个来源均已成功时整条 workflow 跳过。
- 手动 `Run workflow`：强制刷新两个来源并执行完整下游流程。

旧的 OpenAlex、DeepSeek 摘要和邮件 workflow 保留为手动维护入口，不再单独定时运行。

## 统一发现与查重

两个来源先写入不可变的原始审计库：

- `data/paper_registry.jsonl`

随后 `scripts.pipeline.reconcile_registry` 生成：

- `data/unified_paper_registry.jsonl`：所有下游步骤唯一读取的统一候选库；
- `state/unified_candidate_aliases.json`：被合并 candidate ID 到 canonical ID 的映射；
- `state/unified_registry_manifest.json`：数量、来源组合和文件哈希。

跨来源身份按 DOI、OpenAlex work ID、规范化标题与年份、原始内容指纹依次合并，并使用保守的同标题、年份与第一作者兼容规则补充无 DOI 情况。已写入完成历史的 candidate ID 优先保留；否则 Scholar ID 优先，以保留 Scholar 的期刊 mandatory 策略。每条统一记录保存 `source_provenance`，原始来源记录不删除。

## 统一下游链路

只有两个来源在当天都记录成功后，才进入：

```text
统一 registry
→ Crossref/OpenAlex enrichment
→ 项目识别与路由
→ 确定性评分和每日预算
→ DeepSeek Pro 中文摘要
→ 本地结构、中文、方法深度、数字和架构校验
→ 自动写入 summary_history
→ 长期知识库
→ 每日一封汇总邮件
```

若上午某来源失败，下游不会在不完整数据上运行；晚间只补跑失败来源。若晚间仍失败，诊断状态会持久化，workflow 显示失败，不发送不完整日报。

## 自动摘要事务

`completed_candidate_ids` 在任何模型调用前被过滤。每天最多生成三个正式摘要。公开全文方法内容只在 runner 临时使用；无法获得时回退标题、元数据和摘要。

只有整批全部通过本地校验才会写入 `state/summary_history.json`。失败或部分完成的批次不写完成历史，候选会在以后重试。生产链路不需要人工评审。

## 长期知识库

每篇自动完成的论文幂等写入：

- `data/knowledge_base/papers.jsonl`：论文、中文方法摘要、标签和搜索文本；
- `data/knowledge_base/index.json`：按 candidate ID、DOI、项目、年份、期刊、日期和方法标签索引；
- `data/knowledge_base/index.md`：可读的项目索引；
- `state/knowledge_base_manifest.json`：计数与 SHA-256。

不保存整篇正文或正文摘录。

## 每日邮件

完整流程结束后向配置 allowlist 中的 `a209072780@126.com` 发送一封中文日报：

- 有摘要：发送完整中文摘要、方法说明和 DOI 链接；
- 没有论文进入摘要名额：仍发送零篇日报，并明确本次没有产生模型 token；
- 摘要被选中但生成或校验失败：不发送伪装成空日报的邮件，workflow 失败并等待重试。

邮件使用确定性 RFC Message-ID 和 `state/email_delivery_state.json` 双重幂等保护，重跑不会重复发送。Gmail OAuth 只需要 `gmail.readonly` 和 `gmail.send`。
