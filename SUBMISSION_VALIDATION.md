# VoiceOps 2026 - Submission validation record

This document records the verification gates run for the speech-verified submission candidate. The single hidden-case-dependent test is excluded because its private input is intentionally not packaged.

## Offline validation

- Python compilation: passed (exit code 0).
- Public-package pytest command: `69 passed, 1 deselected` (exit code 0).
- The deselected test is `tests/test_retrieval_v8.py::test_hidden_case_evidence_recall_at_seven_is_complete`; it requires `eval/hidden_cases.csv`, which is intentionally excluded. The unfiltered full suite is not claimed to pass.
- Dataset validator: passed (exit code 0; 9 CSVs, 21 knowledge-base documents, 1,000 orders, 1,783 items, 244 logs, 52 questions, and 23 conversations).
- Offline answer-quality gate: passed for the recorded normal and ASR-text question evaluations.
- Offline normal questions: 52 items, 100.0% content accuracy, 100.0% decision accuracy, mean retrieval recall 0.968.
- Offline ASR-text questions: 52 items, 100.0% content accuracy, 100.0% decision accuracy, mean retrieval recall 0.968.
- Offline conversations: 96 turns across 23 conversations, 100.0% checks/decision/tool-call accuracy, 23/23 conversations fully correct, mean retrieval recall 0.944.

No evaluation was rerun for this speech diagnostic update; the retrieval and answer metrics above remain the previously recorded candidate results.

## Recorded bounded RAG evaluations

| Evaluation | Content / checks | Decision | Mean retrieval recall |
|---|---:|---:|---:|
| Normal questions (52) | 98.1% content | 96.2% | 0.968 |
| ASR-text questions (52) | 100.0% content | 94.2% | 0.968 |
| Conversations (96 turns) | 100.0% checks | 100.0% | 0.944 |

For conversations, 23/23 were fully correct in the recorded run. An earlier ASR RAG attempt ended in a provider transport error and offline fallback; that attempt is not treated as a valid RAG result.

## Known limitations

- Offline retrieval recall is below the saved baseline: normal 0.989 to 0.968; ASR 0.989 to 0.968; conversations 1.000 to 0.944. The tested HI_MAP and context-query rollback experiments did not recover the decrease.
- The latest bounded ASR RAG evaluation completed but had 94.2% decision accuracy.
- Latest bounded synthetic-clip smoke test: STT returned a nonempty transcript with exactly one request; TTS produced a readable, nonempty WAV with exactly one request. The transcript text and credential values were not printed.
- The earlier STT failure was not reproduced. Its historical root cause is not established, and no fix to that failure is claimed.
- Browser microphone behavior and complete production end-to-end speech integration were not tested.
- The held-out diagnostic set was highly templated and was not treated as reliable evidence of general retrieval quality.
- Embedding retrieval was not validated with a confirmed embedding model.

## Package audit

The rebuilt archive `voiceops_2026_submission_verified_speech_20261029_r1.zip` passed ZIP CRC validation. All 46 manifest-listed payload files matched their recorded sizes and SHA-256 values, and archive membership matched the manifest exactly. Required public files were present; datasets, hidden cases/labels, evaluation-item JSONs, caches, request-budget ledgers, secrets, logs, backups, temporary audio, bytecode, and virtual environments were absent. Packaged source and documentation matched the reviewed branch checkout. The report PDF is readable and rendered as a clean two-page document.

The included `demo.mp4` is a previously recorded synthetic asset; it is not itself proof of a passing STT test. A separate bounded synthetic-clip smoke test passed STT and TTS as recorded above.

## Repositories

- Repository: https://github.com/chukka-venugopalam/voiceops
- Submission branch: https://github.com/chukka-venugopalam/voiceops/tree/submission/voiceops-final-20261029
- Pull request: https://github.com/chukka-venugopalam/voiceops/pull/1
