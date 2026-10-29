# VoiceOps 2026 — Submission Validation Notes

This file records the local offline gate results for the source included in this package. It does not claim a competition rank or guarantee a winning score.

## Finalization gates

- Python compileall exit code: `0`
- Full pytest exit code: `0`
- Pytest summary: `62 passed in 30.46s`
- Dataset validator exit code: `0`
- Offline content/decision accuracy gate: `PASS`
- No LLM, embedding, Gnani, STT/TTS, or live-service calls were made by this finalization tool.
- `.env`, `.env.example`, dataset contents, evaluation-item text, and request-budget ledgers were not packaged or inspected by this tool.

## Offline metrics from the selected candidate

| Evaluation | Content / decision / checks | Retrieval recall | Source |
|---|---|---:|---|
| Normal questions | content 100.0%; decision 100.0% | 0.968 | `v9_1_2_ablation_no_hi_map_no_context_questions.json` |
| ASR questions | content 100.0%; decision 100.0% | 0.968 | `v9_1_2_ablation_no_hi_map_no_context_asr.json` |
| Conversations | checks 100.0%; tool calls 100.0%; fully correct 23/23/23 | 0.944 | `v9_1_2_ablation_no_hi_map_no_context_conversations.json` |

## Known limitation

The verified pre-v9.1.2 offline summaries reported normal/ASR recall of `0.989` and conversation recall of `1.000`. The selected candidate's recall was lower in the measured ablations, while answer accuracy and conversation correctness remained unchanged. This package does not claim a retrieval-recall improvement. The held-out set used previously was highly templated and is not used as strong evidence for ranking improvements.

A prior ASR LLM/RAG evaluation ended with a provider transport error and offline fallback, so that run is not reported as a valid RAG result. No new LLM/RAG run was made here. Embedding retrieval remains unvalidated because a suitable embedding model was not confirmed.

## Included assets

- `report.pdf` and `demo.mp4` are included for submission completeness. If sourced from the previous submission archive, they are carried-forward assets and were not regenerated in this finalization run.
- Public source repository: https://github.com/chukka-venugopalam/voiceops

## Reproducibility

The submission ZIP includes a file manifest with sizes and SHA-256 hashes. The evaluation result JSONs themselves, datasets, hidden cases, held-out labels, caches, backups, and credentials are intentionally excluded.
