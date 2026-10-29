# Release notes - 2026-10-29

This archive contains the tested VoiceOps 2026 submission candidate after two controlled retrieval rollback experiments. Those rollbacks did not restore the previous recall values. No retrieval or answer-quality experiments were performed for this speech verification update.

## Recorded validation

- Python compilation passed.
- Public-package pytest gate: 69 passed, 1 deselected. The deselected test requires the intentionally excluded `eval/hidden_cases.csv`; the unfiltered suite is not reported as passing.
- The dataset validator passed.
- Offline questions: 100.0% content and decision accuracy on 52 normal and 52 ASR-text items; recall 0.968 for each.
- Offline conversations: 100.0% checks and 23/23 fully correct conversations across 96 turns; recall 0.944.
- Latest bounded RAG results: normal 98.1% content / 96.2% decision; ASR-text 100.0% content / 94.2% decision; conversations 100.0% checks and 23/23 fully correct.
- Latest bounded synthetic-clip smoke test: STT returned a nonempty transcript with one request; TTS produced a readable nonempty WAV with one request.
- The earlier STT failure was not reproduced, and its historical root cause is not established. Browser microphone behavior and complete production end-to-end integration remain unverified.

## Limitations carried into this candidate

Retrieval recall is below the saved previous baseline (normal/ASR 0.989 to 0.968; conversations 1.000 to 0.944). The latest ASR RAG decision accuracy was 94.2%. A prior ASR RAG attempt that had a provider transport error and offline fallback is invalid as RAG evidence. The held-out diagnostic set was highly templated and is not used as robust evidence of retrieval quality. These limitations are disclosed in `report.pdf` and `SUBMISSION_VALIDATION.md`.

## Package integrity and links

The archive includes `submission_manifest.json` with sizes and SHA-256 hashes for all other packaged files. Private dataset files, hidden cases/labels, result JSONs, caches, request-budget ledgers, credentials, logs, backups and temporary audio are excluded.

- Repository: https://github.com/chukka-venugopalam/voiceops
- Submission branch: https://github.com/chukka-venugopalam/voiceops/tree/submission/voiceops-final-20261029
- Pull request: https://github.com/chukka-venugopalam/voiceops/pull/1


## Speech integration diagnostic update

- Added `scripts/gnani_smoke_test.py`: converts the included prerecorded synthetic demo audio to mono 16-bit PCM at 16 kHz, validates duration/signal level, runs bounded STT and optional TTS checks with zero retries, and prints sanitized provider diagnostic fields without printing transcript text or credential values.
- Added `--stt-only` and `--tts-only` modes so either speech operation can be checked without repeating the other provider request.
- Added offline tests for audio normalization, WAV inspection, credential redaction, exception-cause retention, sanitized errors, and isolation of STT-only/TTS-only modes.
- Updated `preflight.py --gnani` to invoke the bundled bounded diagnostic and surface its sanitized output.
- Updated the STT wrapper to preserve the provider exception as the cause for safe diagnostic inspection; raw exception messages are not emitted by the diagnostic.
- The one bounded STT check and one bounded TTS check passed on the bundled synthetic clip. This does not establish a fix for the historical STT failure or verify browser/production end-to-end behavior.
