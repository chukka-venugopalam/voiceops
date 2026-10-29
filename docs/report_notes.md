# Evaluation notes

## Measured outcomes

- Offline normal and ASR-text question evaluations each achieved 100.0% content and decision accuracy over 52 items, with mean retrieval recall 0.968.
- The offline 96-turn conversation run reported 100.0% checks and 23/23 fully correct conversations, with mean retrieval recall 0.944.
- Latest bounded RAG evaluations reported 98.1% content / 96.2% decision for normal questions; 100.0% content / 94.2% decision for ASR-text questions; and 100.0% checks with 23/23 fully correct conversations.

## Known limitations

The recall values remain below the verified saved baseline. Reverting the HI_MAP additions and the context-query call-site change did not recover those values. The diagnostic held-out set was highly templated and should not be taken as robust evidence for general retrieval quality. A previous ASR RAG attempt suffered a provider transport error and offline fallback; its output is not valid RAG evidence.

The latest bounded smoke test on prerecorded synthetic audio passed: STT returned a nonempty transcript with one request, and TTS generated a readable nonempty WAV with one request. The earlier STT failure was not reproduced; its historical root cause is not established, and no fix to it is claimed. Browser microphone behavior and complete production end-to-end speech were not tested. Embedding retrieval was not validated using a confirmed model configuration.

The public-package pytest command passed with 69 tests and one deselected hidden-case-dependent test. That test needs `eval/hidden_cases.csv`, which is intentionally excluded from the repository and submission package. Do not describe the unfiltered suite as passing. Compilation and the dataset validator passed.

## Follow-up work

Investigate generic retrieval ranking and query normalization against the verified baselines; the prior STT failure remains unexplained and should be investigated only if reproduced; and broaden Hindi/Hinglish evaluation using synthetic, non-sensitive speech. Any future performance claim should refer to the exact evaluation mode and item counts used.
