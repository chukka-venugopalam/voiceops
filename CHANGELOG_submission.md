# Submission changelog

- Added robust matching for catalog titles containing letter-digit model names and context-aware spoken storage sizes, plus current electronics-return evidence ranking.
- Retained time-based ranking for historical/superseded policies; ambiguous color variants are not inferred from queries that omit color.
- Hardened Gnani WAV handling for sentinel-sized RIFF length fields; the synthetic 3-turn STT/TTS run completed with zero errors.
- Runner refuses to overwrite pre-existing output files, reports per-file failures with a nonzero process exit, and honors `VOICEOPS_NO_ENV_LOAD=1` without reading `.env`.
- Verified 47/47 tests, 23/23 focused product/retrieval tests, dataset validation, offline evaluation summaries, and the 3-turn voice output.
- Verified saved baseline steps: Step 1 prompt-only (23.1% content, 88.5% decision) and Step 2 RAG (96.2% content/decision, 0.819 retrieval recall); this sprint made no new LLM calls.
- Public target repository: https://github.com/chukka-venugopalam/voiceops
- The archive intentionally excludes `eval/hidden_cases.csv`; the hidden-case retrieval test needs that separately retained evaluation file to run.
