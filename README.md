# VoiceOps 2026 - ZentroMart voice support agent

A customer-support agent for the fictional store ZentroMart. It answers from the company documents, holds a multi-turn
conversation (identity check, order lookup, tickets, handover to a human), and speaks and listens with Gnani STT/TTS.

```
customer audio -> Gnani STT -> [ agent: language + intent -> retrieval (policy sections) -> code facts (dates, deadlines,
                  escalation rules) -> LLM wording -> guardrails ] -> Gnani TTS -> spoken reply
```

## 1. Set up (10 minutes)
```bash
# Python 3.9+.  Unzip this folder, open a terminal inside it.
python -m venv .venv                      # Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# Create a local .env with the variable names listed below; credentials are not distributed.
```
Edit `.env`:
| Variable | What to put |
|---|---|
| `DATASET_DIR` | Path to the extracted `Capstone_Project_Dataset_FTTT_2026` folder (the one containing `kb/`, `data/`, `tools/`). The dataset is **not** part of this repo. |
| `GNANI_API_KEY` | Key from app.gnani.ai/voice (covers STT and TTS). |
| `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL` | Any hosted LLM: `anthropic`, `openai`, `groq`, `gemini`, `openrouter`, or `openai_compat` + `LLM_BASE_URL`. Set the model name your key can use. |

Then: `python scripts/preflight.py` (checks everything; add `--ping` for one tiny LLM call, `--models` to list the model ids your LLM key can use, `--gnani` for the Gnani test).

**Gemini free tier notes (checked Oct 2026):** Gemini 2.5 models were retired; use a current Flash / Flash-Lite model (`LLM_PROVIDER=gemini`, default `gemini-3.1-flash-lite`).
Google no longer publishes free-tier limits: look at your own per-model limits at https://aistudio.google.com/rate-limit before running the evaluations (a full first pass needs roughly 250-400 calls).
Set `LLM_MIN_INTERVAL=4.5` to stay under about 13 calls per minute. If a model's daily quota is exhausted the evaluation stops and says so (it never silently falls back to offline answers); answers already received are cached.
Free-tier requests may be used by Google to improve its products: that is fine for this synthetic dataset, never send real customer data.

## 2. Work in this order (saves credits)
| # | Command | What it does | Uses Gnani credits? |
|---|---|---|---|
| 1 | `python -m pytest -q` | 62 unit/regression tests | no |
| 2 | `python chat.py -v` | Talk to the agent in text. `--offline` forces rules-only mode | no |
| 3 | `python eval/eval_questions.py --mode prompt_only` | **Step 1** score: LLM with only a role prompt | no |
| 4 | `python eval/eval_questions.py --mode rag` | **Step 2** score: full agent (compare with #3 for your report) | no |
| 5 | `python eval/eval_questions.py --mode rag --field asr_text` | Same, on speech-recognition-style text | no |
| 6 | `python eval/eval_conversations.py --mode rag` | **Step 3** score on the 23 practice conversations (96 turns) | no |
| 7 | Record ~10 short clips of **your own voice**, put them in `my_audio/`, then `python run.py --audio-dir my_audio --output-dir out` | End-to-end voice workflow | yes |
| 8 | `python demo_server.py` -> open http://localhost:8000 | Live voice demo page | yes |

LLM responses are cached in `.cache/` (never submit that directory). Do development and evaluation on text before using audio. Evaluation runs write result JSON files under `eval/results/`; those generated results and any hidden or held-out evaluation inputs are not included in this repository.

For strictly offline checks that must not read `.env`, set `VOICEOPS_NO_ENV_LOAD=1` and supply `DATASET_DIR` and `TICKETS_PATH` explicitly in the environment. This guard is supported by the test and offline evaluation entry points.

## 3. The organizers' command
```bash
python run.py --audio-dir <folder of customer audio> --output-dir <output folder> --language-code en-IN
```
For each audio file it writes `<name>.reply.txt`, `<name>.reply.wav`, `<name>.json` (transcript, reply, decision, retrieved section IDs, tool calls)
and `results.csv` / `results.jsonl`. A bad file never stops the batch. Files named like `PC003_t1.wav, PC003_t2.wav` (or a `manifest.csv`
with `file,conversation_id,turn`) are treated as one call; otherwise every file is its own call.
`--dry-run` tests the whole pipeline with no Gnani calls (reads `<file>.txt` as the transcript, writes silent wavs). `--questions practice/questions.csv --field asr_text`
runs text only.

## 4. What the code does (so you can explain it)
- **Identity (IDV-1..4):** order details only after the caller gives the last 4 phone digits or the delivery pincode; one retry; then refuse. Never reads out phone, email, address or pincode.
- **Escalation (ESC-1..4):** human request, legal/consumer court, safety risk, activity the customer did not make, 3rd+ contact (`contacts_last_7_days >= 2`), refund >3 business days past its maximum, damage >Rs 10,000 -> `create_ticket` first, then tell the customer a colleague will contact them. "Ticket created" is said only if `created=True`.
- **Dates/rules in code (`calc.py`, `facts.py`):** refund ETA and overdue check (Mon-Sat minus `holidays.csv`), return windows (7/10/15/30-day rules), pickup fee, COD limit, delivery fee. The LLM is told to trust these.
- **Retrieval (`kb.py`, `agent.py`):** section-level chunks with status and date; BM25 ranked by document precedence; LLM query rewrite; topic/order-aware routing of key sections; catalog rows for products named. The reply records the section IDs it used.
- **Offline mode:** with no LLM key the agent still runs (rules + quoting policy sentences). Useful as a floor and for testing; a real LLM gives much better wording and Hindi/Hinglish.
- **Voice (`voice.py`):** Gnani STT (selected language, en-IN by default), long-clip splitting (REST limit 60 s), ffmpeg conversion for unsupported formats, caching, retries, and correction of sentinel-sized RIFF/WAV length fields; replies are spoken with month-name dates, order IDs in groups of four, amounts in words.

## 5. Honest status (read before you trust the numbers)
- **Step 1 baseline:** a saved `prompt_only` result is available (52 questions: 23.1% content accuracy, 88.5% decision accuracy, 0.0 retrieval recall). This is distinct from the rules-only offline evaluations below.
- Current offline validation: 62 tests passed and the dataset validator passed. On 52 normal questions and 52 ASR-text questions, content and decision accuracy were 100%; retrieval recall was 0.968 for each. The 96-turn conversation evaluation passed all checks with 23/23 conversations fully correct; retrieval recall was 0.944.
- Compared with the verified saved baseline, retrieval recall is lower: normal 0.989 → 0.968, ASR 0.989 → 0.968, and conversations 1.000 → 0.944. The HI_MAP and context-query rollback experiments did not recover the decline.
- Latest bounded LLM/RAG observations: normal 98.1% content, 96.2% decision, 0.968 recall; ASR 100% content, 94.2% decision, 0.968 recall; conversations 100% checks and 23/23 fully correct, with 0.944 recall. An earlier ASR RAG attempt ended in a provider transport error and offline fallback, so that run is invalid RAG evidence.
- Gnani smoke test: TTS generated a non-empty synthetic WAV on the retry; the single STT attempt failed. Speech integration testing is incomplete. The previously recorded synthetic demo does not establish that the current STT test passed.
- **Release status: INCOMPLETE_INTEGRATION_TESTS.** These metrics do not guarantee a competition score. Browser microphone behavior and the user's own recordings have not been verified.
- Unknown until you ask the organizers: how multi-turn conversations are fed in the hidden test (we support single files and `_t1/_t2` naming), and which language each test audio is in (we try en-IN, then hi-IN).
- The 30 additional cases used for local diagnosis were written with knowledge of the documents, so their scores are optimistic. Those cases are not included in this repository.

Release notes: see [CHANGELOG_submission.md](./CHANGELOG_submission.md).

## 6. Troubleshooting
- `Dataset not found`: set `DATASET_DIR` in `.env` to the folder that contains `kb/` and `data/`.
- `GNANI_API_KEY is not set`: edit `.env` (no quotes needed). Use `--dry-run` to test without it.
- STT error on a recording: install ffmpeg (browser recordings are webm) or record wav/mp3.
- LLM errors: the agent falls back to offline rules for that turn and keeps going; check `LLM_MODEL` spelling with `python scripts/preflight.py --ping`.
- Handover tickets are written by the organizers' `tools.py` to `tools/tickets_created.csv` inside the dataset folder (our tests and evals redirect them to a temp file).
- Never commit `.env` or credentials. Keep local configuration and generated evaluation data out of version control.

## 7. Layout
```
run.py  chat.py  demo_server.py        entry points (batch run / text chat / live demo page)
src/voiceops/   agent.py (conversation + escalation)  facts.py calc.py (exact rules)  kb.py (chunks + retrieval)
                llm.py (any provider)  voice.py (Gnani)  intent.py textnorm.py speech_text.py  scorer.py evalkit.py
eval/           eval_questions.py  eval_conversations.py  heldout_retrieval_eval.py
scripts/        preflight.py  sanitize_eval_results.py
docs/           CHECKLIST.md  DEMO_SCRIPT.md  report_notes.md
```

## Source repository

https://github.com/chukka-venugopalam/voiceops
