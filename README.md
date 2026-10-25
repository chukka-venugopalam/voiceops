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

Edit `.env` locally: set `DATASET_DIR` to the dataset folder and set `GNANI_API_KEY` for real speech processing. Do not share or commit the key. No LLM key is needed for offline mode; `--offline` uses the rules-based agent and makes no LLM calls. Audio input still uses Gnani STT/TTS unless `--dry-run` is used.

## Audio-folder runner

```powershell
python run.py --audio-dir "samples\incoming" --output-dir "output" --language-code en-IN --offline
```

The runner accepts `--output-dir` (also `--out`) and `--language-code`. For each audio input it writes `<name>.reply.txt`, `<name>.reply.wav`, and `<name>.json`, including transcript, reply, decision, evidence IDs, tool calls, and errors; it also writes `results.csv` and `results.jsonl`. The output directory must not already contain output files with those names; the runner refuses to overwrite them. If a file fails, the batch continues and exits nonzero after writing its error record. Empty transcripts receive a clarification reply and spoken audio. `--offline` disables external LLM calls but still uses Gnani STT/TTS for audio, so configure `GNANI_API_KEY` locally. `--dry-run` reads a same-name `.txt` transcript and makes silent reply audio without Gnani.

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
- The verified synthetic voice run used real Gnani STT/TTS with the agent in offline rules mode. No personal recordings were used.

Browser microphone behavior, personal recordings, and new live LLM behavior were not tested. Previous saved RAG scores are reported as existing baselines; this sprint made no external LLM calls.
