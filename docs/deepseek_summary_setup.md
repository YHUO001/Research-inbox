# DeepSeek summary provider setup

The retrieval, parsing, recognition, routing, enrichment, and scoring stages remain deterministic. DeepSeek is used only for the final, manually triggered summary stage.

## Provider policy

- Model: `deepseek-v4-pro`
- API endpoint: `https://api.deepseek.com/chat/completions`
- Thinking mode: disabled
- Response format: JSON object
- Maximum papers per run: 3
- Input basis: title, metadata, abstract, project labels, and deterministic score evidence
- Full text: disabled
- Email delivery: disabled
- Summary history update: disabled during provider validation
- Scheduled model calls: disabled

The production summary model is Pro because the repository benchmark showed materially better schema and technical-detail adherence than Flash at the current three-paper daily budget. The benchmark workflow retains both models for future comparisons; changing the production model does not make retrieval, recognition, routing, enrichment, or scoring dependent on an LLM.

DeepSeek JSON mode guarantees parseable JSON, not conformance to the repository JSON Schema. Every response is therefore checked locally against `schemas/paper_summary.schema.json` and rejected when it contains unsupported numerical claims.

## GitHub secret

Create a repository Actions secret named:

```text
DEEPSEEK_API_KEY
```

Do not put the key in repository files, workflow inputs, logs, issues, or pull requests.

## Manual validation run

1. Run `Build Summary Dry Run` to refresh the provider-neutral request package.
2. Run `Generate DeepSeek Summaries`.
3. Inspect:
   - `data/summaries/YYYY-MM-DD.jsonl`
   - `data/digests/YYYY-MM-DD.generated.json`
   - `data/digests/YYYY-MM-DD.generated.md`
   - `state/summary_generation_manifest.json`

The workflow persists validation diagnostics before reporting a failure. It does not use Gmail credentials and cannot send email.

## Cost accounting

The manifest records returned token usage and estimates cost in CNY using the Pro prices declared in `config/summary_generation.yaml`. The estimate is auditable but the DeepSeek account billing page remains the billing source of truth.
