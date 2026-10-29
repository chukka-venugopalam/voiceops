# Submission candidate — 2026-10-29

## Status

`INCOMPLETE_INTEGRATION_TESTS`. This is a locally validated submission candidate, not a guaranteed competition result.

## Validation

- Python compilation passed.
- Full test suite: 62 passed.
- Dataset validation passed.
- Offline normal and ASR evaluations: 100% content and decision accuracy; retrieval recall 0.968 on each.
- Offline conversations: 100% checks, 23/23 fully correct; retrieval recall 0.944.
- Latest bounded RAG evaluations: normal 98.1% content, 96.2% decision, 0.968 recall; ASR 100% content, 94.2% decision, 0.968 recall; conversations 100% checks and 23/23 fully correct, 0.944 recall.
- Gnani TTS generated a non-empty synthetic WAV on the retry. The single STT smoke-test attempt failed; speech integration testing is incomplete.

## Known limitations

Compared with verified saved offline baselines, recall remains lower: normal 0.989 → 0.968, ASR 0.989 → 0.968, conversations 1.000 → 0.944. The completed HI_MAP and context-query rollbacks did not recover the decrease. An earlier ASR RAG run ended with a provider transport error and offline fallback and is invalid RAG evidence. The latest ASR RAG result completed but recorded 94.2% decision accuracy.

The held-out retrieval set used during diagnosis was highly templated and should not be treated as robust evidence for retrieval quality.

## Candidate contents and privacy

The submission package excludes `.env`, `.env.example`, credentials, the private dataset, hidden cases, held-out labels, raw evaluation results, caches, request-budget ledgers, temporary audio, logs, and backups. The previously recorded synthetic demo does not prove that the current STT test passed.

## Reproducibility

Use the project Python virtual environment, install dependencies from `requirements.txt`, configure credentials locally when needed, and run:

```text
python -m pytest -q
python <DATASET_DIR>\validate_dataset.py
```

For offline checks, set `VOICEOPS_NO_ENV_LOAD=1`, `VOICEOPS_ENABLE_HYBRID_RAG=0`, and `DATASET_DIR` to the dataset root.

Repository: https://github.com/chukka-venugopalam/voiceops
