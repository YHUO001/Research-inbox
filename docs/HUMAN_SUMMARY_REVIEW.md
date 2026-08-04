# 中文论文摘要与人工评审流程

摘要层与检索、识别、元数据补全、路由和确定性评分保持分离。正式摘要默认使用简体中文，并把“方法原理”和“具体实施过程”作为核心内容。

## 正文使用边界

生成阶段会按以下顺序处理每篇论文：

1. 使用标题、元数据和摘要作为基础证据；
2. 尝试访问 `open_access_url`、论文落地页或 DOI 页面；
3. 仅从公开可访问的 HTML/PDF 中提取 Methods、Experimental setup、Implementation、Training、Fabrication 等方法相关内容；
4. 提取内容只在当次 Actions runner 中临时使用，不提交到仓库；
5. manifest 只保存来源 URL、媒体类型、章节标题、字符数和内容哈希；
6. 公开正文不可获得时自动退回摘要级总结。

公开正文只用于定性解释方法原理和实施流程。所有数字仍必须出现在标题或摘要中，避免正文解析错误引入难以核查的数值。

## 正式摘要结构

每篇摘要至少包含：

- 核心问题；
- 方法概览与架构；
- 方法原理：解释各组成部分如何协同、方法为什么能够工作；
- 具体实施过程：2–6 个完整段落，按顺序说明输入、核心操作、训练或参数配置、硬件/软件执行以及输出产生过程；
- 主要贡献；
- 作者报告的结果；
- 与既有工作的区别；
- 研究价值；
- 局限与开放问题；
- ONN 或 ZO 专项技术字段。

所有面向读者的叙述必须是中文；论文标题、模型名称、标准缩写和必要的英文术语可以保留。

## 状态转换

```text
通过本地校验的中文摘要
  -> pending_human_review
  -> approved_human_review 或 revision_requested
```

任何阶段都不会自动发送邮件。

`summary_history.json` 只会在 **Finalize Reviewed Summaries** 中明确选择 `approve_all` 后更新。`hold_for_revision` 不会把候选论文标记为完成，因此仍可修订或重新生成。

## 评审材料

运行 **Prepare Human Summary Review** 后，会生成：

- `data/reviews/YYYY-MM-DD.review.md`
- `data/reviews/YYYY-MM-DD.review.json`

评审包包含原始摘要、正文方法来源、中文摘要、自动数字校验、中文校验、方法深度校验和架构证据。

评审重点不是逐项小分，而是确认：

- 事实与数字是否忠实；
- 方法原理是否真正讲清楚；
- 实施过程是否足够具体，能理解作者如何完成工作；
- 技术分类是否合理；
- 是否足以支持研究筛选和后续精读。

## 最终批准

所有论文都达到要求后，运行 **Finalize Reviewed Summaries**：

- decision：`approve_all`
- confirmation：`REVIEWED`
- notes：可填写总体评价

任意一篇需要修改时选择 `hold_for_revision`。当前批次最多三篇，批准采用整批全有或全无规则。
