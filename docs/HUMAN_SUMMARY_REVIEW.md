# Human summary review gate

The summary layer is intentionally separated from retrieval, recognition, enrichment, routing, and deterministic scoring.

## State transitions

```text
validated summaries
  -> pending_human_review
  -> approved_human_review OR revision_requested
```

No email is sent in any of these states.

`summary_history.json` is updated only by an explicit `approve_all` decision in the **Finalize Reviewed Summaries** workflow. A `hold_for_revision` decision does not mark candidates as completed, so they remain available for correction and regeneration.

## Prepare the current review packet

Run **Prepare Human Summary Review**. This workflow:

- uses the existing generated summaries;
- makes no model or provider API call;
- enforces evidence-grounded ONN architecture classification;
- revalidates JSON Schema and numerical grounding;
- writes `data/reviews/YYYY-MM-DD.review.md` and `.review.json` to `automation-state`.

## Review rubric

For each paper, compare the generated summary with the supplied abstract and score 1–5:

1. factual accuracy;
2. completeness for the abstract;
3. technical classification;
4. usefulness for research triage.

Use `revise` when a factual statement, key omission, architecture label, training description, limitation, or research-value claim needs correction.

## Finalize

After every paper is acceptable, run **Finalize Reviewed Summaries** with:

- decision: `approve_all`;
- confirmation: `REVIEWED`;
- optional batch notes.

Choose `hold_for_revision` when any paper needs changes. Approval is all-or-nothing for the current maximum batch of three papers.
