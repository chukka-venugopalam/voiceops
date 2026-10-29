# VoiceOps 2026 - ZentroMart voice support agent

VoiceOps is a voice-capable support agent for the fictional store ZentroMart. It combines section-based document retrieval, deterministic business rules, conversation state, escalation/ticket workflows, optional LLM response wording, and Gnani speech input/output.

**Release status:** the latest bounded synthetic-clip smoke test passed STT (one request, nonempty transcript) and TTS (one request, readable nonempty WAV). The earlier STT failure was not reproduced; its historical root cause is not established. Browser microphone behavior and full production end-to-end integration are not verified. This is a tested submission candidate, not a guarantee of a competition score.

## Repository and submission links

- Repository: https://github.com/chukka-venugopalam/voiceops
- Submission branch: https://github.com/chukka-venugopalam/voiceops/tree/submission/voiceops-final-20261029
- Pull request: https://github.com/chukka-venugopalam/voiceops/pull/1

The submission candidate is on the branch above. The repository's default `main` branch may not show this exact candidate until its pull request is merged.

## Contents

- `src/voiceops/`: agent flow, retrieval, business facts/calculations, language handling, LLM adapter, speech integration.
- `run.py`: batch runner for audio or text inputs.
- `chat.py`, `demo_server.py`: text chat and local demo interface.
- `eval/`: question/conversation evaluators and retrieval diagnostics.
- `tests/`: offline unit and regression tests.
- `report.pdf`: technical submission report.
- `demo.mp4`: previously recorded synthetic demo asset; it is not evidence that the latest STT smoke test passed.
- `submission_manifest.json`: SHA-256 hashes and sizes for all other files in this ZIP.

The capstone dataset is separate and is deliberately not included. It contains the `kb/`, `data/`, `practice/`, and `tools/` directories, plus `validate_dataset.py` at its root.

## Setup (Windows PowerShell)

Use Python 3.9 or newer. Extract the project and install the listed dependencies:

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Set `DATASET_DIR` to the separately extracted dataset root (the folder containing `kb/` and `data/`). Create a local `.env` file only if you need the live provider integrations. The key variables are `DATASET_DIR`, `GNANI_API_KEY`, `LLM_PROVIDER`, `LLM_API_KEY`, and `LLM_MODEL`; use values supported by your accounts and never commit credentials. Offline tests do not need provider credentials.

## Offline validation commands

Run these from the project root after the separate dataset is available:

```powershell
$env:DATASET_DIR = "C:\path\to\Capstone_Project_Dataset_FTTT_2026"
$env:VOICEOPS_NO_ENV_LOAD = "1"
$env:VOICEOPS_ENABLE_HYBRID_RAG = "0"
python -m pytest -q --deselect="tests/test_retrieval_v8.py::test_hidden_case_evidence_recall_at_seven_is_complete"
python "$env:DATASET_DIR\validate_dataset.py"
python eval/eval_questions.py --mode offline
python eval/eval_questions.py --mode offline --field asr_text
python eval/eval_conversations.py --mode offline
```

The public-package test command reports 69 passed and 1 deselected. The deselected test requires `eval/hidden_cases.csv`, a private evaluation input intentionally excluded from the repository and submission package. Do not describe the unfiltered suite as passing. Compilation and the separate dataset validator passed.

## Running the agent

For real speech input/output, configure the dataset, Gnani key, and (if desired) LLM provider locally. Provide a folder of supported audio files and use a fresh output directory:

```powershell
python run.py --audio-dir "C:\path\to\audio" --output-dir "out" --language-code en-IN
```

The runner accepts `--out` as an alias for `--output-dir`. Real audio runs call the speech provider; LLM-backed answers may also make provider requests. Do not use real customer data for demonstrations or tests unless you have authorization and appropriate safeguards. The optional `--dry-run` mode reads same-named `.txt` transcripts and writes silent WAVs, without Gnani calls.

The local demo interface can be started with `python demo_server.py`. Browser microphone permissions and live browser-to-agent behavior were not tested; the prerecorded synthetic-file smoke test does not verify that path.

### Gnani speech smoke test

Install dependencies with `python -m pip install -r requirements.txt`. Run `python scripts/gnani_smoke_test.py --stt-only` for one STT request, or `python scripts/gnani_smoke_test.py --tts-only` for one TTS request without repeating STT. Running without either option tests both with at most one request each; live checks use provider credits. The latest bounded synthetic-clip check passed both: STT returned a nonempty transcript (text is never printed), and TTS generated a readable nonempty WAV. The previous STT failure was not reproduced, and its root cause is not established. Browser microphone behavior and complete production end-to-end behavior remain unverified. The script requires `ffmpeg`, normalizes STT input to mono 16-bit PCM at 16 kHz, and reports only sanitized diagnostic fields (not raw provider messages, credentials, or transcript text). `python scripts/preflight.py --gnani` invokes the combined diagnostic. Live checks are not run automatically by pytest or CI.

## Recorded evaluation results

| Evaluation | Recorded result | Mean retrieval recall |
|---|---|---:|
| Offline normal questions (52) | 100.0% content; 100.0% decision | 0.968 |
| Offline ASR-text questions (52) | 100.0% content; 100.0% decision | 0.968 |
| Offline conversations (96 turns / 23 conversations) | 100.0% checks; 23/23 conversations fully correct | 0.944 |
| Bounded RAG normal questions (52) | 98.1% content; 96.2% decision | 0.968 |
| Bounded RAG ASR-text questions (52) | 100.0% content; 94.2% decision | 0.968 |
| Bounded RAG conversations (96 turns) | 100.0% checks and decisions; 23/23 fully correct | 0.944 |

These are the recorded results for this candidate, not independently reproduced by the packaging step. The offline retrieval recall is lower than the saved previous baseline (normal/ASR: 0.989 to 0.968; conversations: 1.000 to 0.944). The tested rollback experiments did not recover the decrease. The latest ASR RAG decision accuracy was 94.2%. A prior ASR RAG run suffered a provider transport error and fell back to offline answers; it is not counted as valid RAG evidence.

## Known limitations

- The earlier STT smoke-test attempt failed, but that failure was not reproduced by the latest bounded test. Latest synthetic-clip STT and TTS checks passed with one request each. No root cause for the earlier failure is known; browser microphone and production end-to-end behavior remain unverified.
- Embedding retrieval was not validated with a confirmed model configuration.
- The held-out diagnostic set was highly templated, so it is not strong evidence of general retrieval quality.
- The existing demo is prerecorded synthetic media; its contents alone do not establish provider verification. The separate bounded smoke-test result is recorded above.

No dataset files, hidden cases/labels, evaluation result JSONs, caches, request-budget ledgers, `.env`, `.env.example`, credentials, `.venv`, `.git`, logs, backups, or temporary audio files are included. See `SUBMISSION_VALIDATION.md` for package and validation details.
