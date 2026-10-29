# Submission checklist

## Build
- [ ] `python scripts/preflight.py --ping --gnani` shows READY
- [ ] Step 1 score recorded: `python eval/eval_questions.py --mode prompt_only`
- [ ] Step 2 score recorded: `python eval/eval_questions.py --mode rag` (must beat Step 1)
- [ ] Conversations: `python eval/eval_conversations.py --mode rag`; read the failing turns and fix real bugs (rules, not single answers)
- [ ] Hidden-style cases: `--file eval/hidden_cases.csv`
- [ ] Speak 10 clips yourself (English, Hinglish, Hindi, an order ID, noise, silence, a handover request) and run `run.py` on them
- [ ] `python run.py --audio-dir my_audio --out sample_outputs` (real agent output only - never edit by hand)

## Deliverables
- [ ] Public GitHub repo with README, `.env.example` (no keys), `requirements.txt`, `run.py`, `src/`
- [ ] `docs/report_notes.md` written honestly, then `python scripts/build_report.py --team "Name" --members "A, B"` -> `report.pdf` (max 4 pages)
- [ ] `demo.mp4` max 2 minutes: one grounded answer with a follow-up + one handover or "I don't know" (see `docs/DEMO_SCRIPT.md`)
- [ ] `python scripts/make_submission.py --team your_team_name` -> `your_team_name_voiceops.zip`

## Final checks (do not skip)
- [ ] Fresh clone in a new folder: `pip install -r requirements.txt`, set `.env`, run the one command on a few audio files
- [ ] Zip has no `.env`, no keys, no `.venv`, no dataset copy (the packager checks)
- [ ] Links on Unstop are public (GitHub, any demo link)
- [ ] Only Gnani used for speech; no other STT/TTS anywhere in the code
- [ ] Submitted a few hours before the deadline (the Unstop card and the banner disagreed about the date: confirm it)
