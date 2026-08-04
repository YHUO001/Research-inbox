# Research Inbox

一个以低 Token 消耗为约束的科研文献发现、筛选、总结与归档项目。

## 当前阶段

项目目前处于配置与数据结构设计阶段。第一阶段不会调用 LLM，也不会读取全文 PDF。

初始目标：

1. 从 Google Scholar Alert 邮件提取新论文候选。
2. 使用 OpenAlex、Crossref 等正式 API 补全元数据。
3. 通过确定性规则和本地相关性评分完成去重与初筛。
4. 只将少量高价值候选交给模型总结。
5. 后续将筛选结果归档到 Zotero。

## 目录结构

```text
.
├── AGENTS.md
├── config/
│   ├── pipeline.yaml
│   ├── research_profile.yaml
│   └── venues.yaml
├── data/
│   ├── normalized/
│   └── raw/
├── logs/
├── schemas/
│   └── paper_record.schema.json
├── scripts/
│   └── README.md
└── state/
    └── pipeline_state.json
```

## 配置文件

- `config/research_profile.yaml`：研究问题、主题、方法、排除主题、种子论文和重点作者。
- `config/venues.yaml`：重点期刊与会议的弱排序信号。
- `config/pipeline.yaml`：数据源、每日处理上限、筛选阈值和 LLM 开关。

## Token 控制原则

- 采集、标准化、去重和初筛优先使用确定性脚本。
- 只处理上次成功运行之后的新数据。
- 没有高分候选时不调用模型。
- 日常流程不读取完整 PDF。
- `config/pipeline.yaml` 中的 `llm.enabled` 默认保持为 `false`。

## 安全原则

不要提交以下内容：

- 邮箱或 API 凭据；
- 原始邮件正文；
- 下载的论文 PDF；
- 私有科研数据；
- 本地数据库和执行日志。

相关路径已经写入 `.gitignore`。

## 下一步

完善 `config/research_profile.yaml`，然后实现 Google Scholar Alert 邮件的增量提取与结构化解析。
