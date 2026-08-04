# DeepSeek Flash vs Pro benchmark — 2026-08-04

> Same three requests, same prompts, JSON mode, and thinking disabled. No email was sent and summary history was not updated.

## Model totals

| Model | Status | Summaries | Prompt tokens | Output tokens | Actual cost (CNY) | All-miss cost (CNY) | Elapsed (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| flash | partial_failure | 0 | 12825 | 5650 | 0.012835 | 0.024125 | 43.29 |
| pro | partial_failure | 2 | 8540 | 3304 | 0.020692 | 0.045444 | 48.90 |

## Deterministic quality indicators

| Model | Valid summaries | Reported results | Contributions | Limitations | not_available | Explicit architecture | Explicit hardware validation |
|---|---:|---:|---:|---:|---:|---:|---:|
| flash | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pro | 2 | 8 | 10 | 6 | 5 | 2 | 2 |

## Interpretation boundary

These indicators verify structure, grounding constraints, coverage, token use, and categorical consistency. They do not by themselves establish that longer or more detailed prose is scientifically better. Final model selection requires paper-by-paper comparison against the supplied abstracts.

## Validation failures

### flash

- `b63d0c87752a364b3611148c`: candidate_id mismatch
- `97f076ce54abbfc51b5b73e4`: candidate_id mismatch
- `766ea277f76b9bc9cd30be5d`: unsupported numeric claims: 5ghz

### pro

- `97f076ce54abbfc51b5b73e4`: unsupported numeric claims: 5.16tops/mm²
