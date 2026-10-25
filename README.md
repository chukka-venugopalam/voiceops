# VoiceOps 2026 - ZentroMart Voice Support Agent

This package contains the VoiceOps source, `report.pdf`, `demo.mp4`, and three synthetic sample turns. The capstone dataset is separate and is not included.

Public repository: https://github.com/chukka-venugopalam/voiceops

## Install and configure

Use Python 3.9 or newer. Extract the supplied dataset folder containing `kb/`, `data/`, `practice/`, and `tools/` separately, then in PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` locally. Set `DATASET_DIR` to the dataset root and `GNANI_API_KEY` for real speech processing. Never share or commit credentials. The application supports these optional LLM settings:

```text
LLM_PROVIDER=<anthropic|openai|groq|gemini|openrouter|openai-compatible>
LLM_API_KEY=<set-locally>
LLM_MODEL=<provider-model-id>
LLM_BASE_URL=<optional-base-url>
```

`LLM_MODEL` is optional when the selected provider has a default model. `LLM_BASE_URL` overrides the provider endpoint and is required for an unlisted OpenAI-compatible endpoint. Instead of `LLM_API_KEY`, the application also recognizes `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`, and `OPENROUTER_API_KEY`. Configure credentials only in your local `.env`; the example above contains placeholders, not usable values.

## Audio-folder runner

### Normal LLM-capable mode

With Gnani speech credentials and an LLM provider/key configured, the runner uses the configured LLM unless `--offline` is specified:

```powershell
python run.py --audio-dir "samples\incoming" --output-dir "output" --language-code en-IN
```

This normal mode can use the configured LLM; no live LLM request was made during this finalization. Speech input and reply audio use Gnani STT/TTS.

### Offline rules-only mode

```powershell
python run.py --audio-dir "samples\incoming" --output-dir "output_offline" --language-code en-IN --offline
```

`--offline` uses deterministic rules instead of the LLM and makes no LLM calls. With real WAV inputs, it still uses Gnani STT/TTS and therefore still needs `GNANI_API_KEY`. For a no-Gnani dry run, provide a same-name `.txt` transcript for each audio file and add `--dry-run`; that mode writes silent reply WAVs.

For each input the runner writes `<name>.reply.txt`, `<name>.reply.wav`, and `<name>.json`, including transcript, reply, decision, evidence IDs, tool calls, and errors. It also writes `results.csv` and `results.jsonl`. The output directory must be fresh: existing output paths are never overwritten. If a file fails, the batch continues and exits nonzero after recording the error. Empty transcripts receive a clarification reply and spoken audio.

The included synthetic conversation demonstrates retrieved evidence IDs such as `RET-2`, `RET-9`, `PLS-2`, and `ESC-1`.

## Verified results

- Step 1 prompt-only baseline (saved): 52 questions, **23.1% content**, **88.5% decision** accuracy, **0.000 retrieval recall**.
- Step 2 RAG baseline (saved): 52 questions, **96.2% content**, **96.2% decision** accuracy, **0.819 retrieval recall**.
- Step 3 RAG conversation baseline (saved): 96 turns, **99.0%** checks/content, **100.0%** decision/tool-call accuracy, **22/23** fully correct conversations.
- Step 4 synthetic voice pipeline: **3 turns**, **3 STT + 3 TTS calls**, **0 errors**; the human-request turn created a handoff ticket.
- Current regression suite: **47 passed**. Focused product/retrieval tests: **23 passed**. Batch runner tests: **4 passed**.
- Test-packaging note: the 47-test run was verified in the full workspace. This clean archive omits the private `eval/hidden_cases.csv` by design, so `test_hidden_case_evidence_recall_at_seven_is_complete` requires that separately retained file to run.
- Dataset validator: **PASS** — 9 CSVs, 21 KB documents, 1,000 orders, 1,783 items, 244 logs, 52 questions, 23 conversations.
- Post-repair rules-only offline questions (52): normal and ASR each scored **100.0% content**, **100.0% decision accuracy**, and **0.989 mean retrieval recall**. These are offline rules metrics, not the Step 1 prompt-only or Step 2 RAG scores.
- Post-repair offline conversations: **96 turns**, **100.0%** checks, decisions, content, and tool-call accuracy; **23/23** fully correct.
- The verified synthetic voice run used real Gnani STT/TTS with the agent in offline rules mode. The demo interleaves the synthetic customer-input clips with their actual synthesized agent replies. No personal recordings were used.

Browser microphone behavior, personal recordings, and live LLM-enabled behavior were not tested in this finalization. Previous saved RAG scores are historical baselines; no external LLM calls were made for this update. Offline scores do not establish live LLM performance.
