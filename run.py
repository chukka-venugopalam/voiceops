#!/usr/bin/env python3
"""BATCH RUN (the command the organizers use).

  python run.py --audio-dir path/to/audio --output-dir out --language-code en-IN

For every audio file: Gnani STT -> agent -> Gnani TTS. Saves, per file, the reply text, the reply audio and the
document chunks used, plus results.jsonl / results.csv. One bad file never stops the batch, but makes the process
exit nonzero after the remaining files have been processed. Existing output files are never overwritten.

Conversations: by default every file is its own call. Files named like PC003_t1.wav, PC003_t2.wav (or a
manifest.csv with columns file,conversation_id,turn in the audio folder) are grouped into one multi-turn call.
Text-only testing (no audio, no Gnani credits):  python run.py --questions practice/questions.csv --field asr_text
Pipeline test without any keys:                  python run.py --audio-dir DIR --dry-run
"""
import argparse, csv, json, os, re, sys, time, traceback, pathlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from voiceops.config import load_env
if os.environ.get("VOICEOPS_NO_ENV_LOAD") != "1":
    load_env()

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac", ".opus"}
GROUP_RE = re.compile(r"^([A-Za-z]+\d+)[_\-](?:t|turn)[_\-]?(\d{1,3})$", re.I)

def plan_sessions(files, folder):
    """Return a list of (conversation_key or None, turn, path) in processing order."""
    manifest = pathlib.Path(folder) / "manifest.csv"
    if manifest.exists():
        rows = {r["file"]: r for r in csv.DictReader(open(manifest, encoding="utf-8"))}
        out = [((rows[f.name]["conversation_id"] if f.name in rows else None), int(rows[f.name]["turn"]) if f.name in rows else 0, f) for f in files]
    else:
        out, groups = [], {}
        for f in files:
            m = GROUP_RE.match(f.stem)
            if m: groups.setdefault(m.group(1), []).append(1)
            out.append((m.group(1) if m else None, int(m.group(2)) if m else 0, f))
        if not any(len(v) >= 2 for v in groups.values()): out = [(None, 0, f) for _, _, f in out]   # no real conversations: stay stateless
    keyed = [x for x in out if x[0]]
    plain = [x for x in out if not x[0]]
    return sorted(keyed, key=lambda x: (x[0], x[1])) + plain

def _output_stem(src, audio_dir):
    return pathlib.Path(str(src)).stem if not isinstance(src, str) or audio_dir else src

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-dir"); ap.add_argument("--out", "--output-dir", dest="out", default="out"); ap.add_argument("--dataset")
    ap.add_argument("--language-code", default="en-IN", help="Gnani STT language code (default: en-IN)")
    ap.add_argument("--tts-model", default="timbre-v2.5")
    ap.add_argument("--tts-voice", default="Kaveri")
    ap.add_argument("--voice-retries", type=int, default=2, help="Retries for Gnani requests")
    ap.add_argument("--voice-cache-dir", default=".cache/voice")
    ap.add_argument("--questions", help="CSV with a text column (practice/questions.csv) instead of audio")
    ap.add_argument("--field", default="asr_text"); ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true", help="no Gnani calls: reads <file>.txt as the transcript, writes silent wavs")
    ap.add_argument("--no-tts", action="store_true"); ap.add_argument("--offline", action="store_true", help="no LLM: rule-based answers")
    args = ap.parse_args()
    if args.dataset: os.environ["DATASET_DIR"] = args.dataset
    from voiceops.agent import Agent
    from voiceops.speech_text import speechify
    from voiceops.voice import Voice
    if args.questions:
        rows = list(csv.DictReader(open(args.questions, encoding="utf-8")))
        items = [(None, 0, r.get("id", f"q{i}"), r[args.field if args.field in r else "question"]) for i, r in enumerate(rows)]
    else:
        if not args.audio_dir: ap.error("--audio-dir or --questions is required")
        files = sorted(p for p in pathlib.Path(args.audio_dir).iterdir() if p.suffix.lower() in AUDIO_EXT)
        items = [(k, t, f, None) for k, t, f in plan_sessions(files, args.audio_dir)]
    if args.limit: items = items[: args.limit]
    out = pathlib.Path(args.out)
    targets = [out / "results.jsonl", out / "results.csv"]
    seen = set()
    for _, _, src, _ in items:
        stem = _output_stem(src, args.audio_dir)
        for suffix in (".reply.txt", ".reply.wav", ".json"):
            target = out / f"{stem}{suffix}"
            key = str(target.resolve()).casefold()
            if key in seen:
                print(f"Refusing to overwrite duplicate output path: {target}", file=sys.stderr)
                return 2
            seen.add(key)
            targets.append(target)
    existing = [target for target in targets if target.exists()]
    if existing:
        print("Refusing to overwrite existing output file(s):", file=sys.stderr)
        for target in existing:
            print(f"  {target}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    agent = Agent(use_llm=not args.offline, channel="voice")
    voice = None if (args.no_tts and args.questions) else Voice(
        dry_run=args.dry_run or bool(args.questions and args.no_tts),
        tts_model=args.tts_model,
        tts_voice=args.tts_voice,
        cache_dir=args.voice_cache_dir,
        retries=args.voice_retries,
    )
    sessions, results = {}, []
    print(f"LLM: {agent.llm.name + ' ' + agent.llm.model if agent.llm_on else 'offline rules'} | {len(items)} item(s)")
    for key, turn, src, text in items:
        stem = _output_stem(src, args.audio_dir)
        rec = {"file": str(getattr(src, "name", src)), "conversation_id": key, "turn": turn, "transcript": "", "reply": "", "decision": "",
               "language": "", "chunk_ids": [], "tool_calls": [], "reply_audio": "", "error": ""}
        t0 = time.time()
        try:
            sess = sessions.setdefault(key, agent.new_session()) if key else agent.new_session()
            if text is None:
                tr = voice.transcribe(str(src), languages=[args.language_code]); text = tr["transcript"]; rec["stt_language"] = tr["language_code"]
            rec["transcript"] = text
            if not text.strip():
                from voiceops.prompts import T
                r = {"reply": T("unclear", sess.lang), "decision": "clarify", "language": sess.lang, "chunk_ids": [], "tool_calls": []}
            else:
                r = agent.respond(sess, text)
            rec.update({k: r[k] for k in ("reply", "decision", "language", "chunk_ids", "tool_calls")})
            (out / f"{stem}.reply.txt").write_text(rec["reply"], encoding="utf-8")
            if voice is not None:
                wav = out / f"{stem}.reply.wav"
                voice.synthesize(speechify(rec["reply"], rec["language"]), rec["language"], str(wav)); rec["reply_audio"] = str(wav)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"; traceback.print_exc(limit=2)
            if not rec["reply"]:
                rec["reply"] = "Sorry, I ran into a problem. Could you please say that again?"
                (out / f"{stem}.reply.txt").write_text(rec["reply"], encoding="utf-8")
                try:
                    if voice is not None:
                        wav = out / f"{stem}.reply.wav"; voice.synthesize(rec["reply"], "en", str(wav)); rec["reply_audio"] = str(wav)
                except Exception as fallback_error:
                    rec["error"] += f"; fallback TTS {type(fallback_error).__name__}: {fallback_error}"
        rec["latency_s"] = round(time.time() - t0, 2)
        (out / f"{stem}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        results.append(rec)
        print(f"[{rec['decision'] or 'ERR':8}] {rec['file']}: {rec['transcript'][:60]!r} -> {rec['reply'][:70]!r}")
    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    with open(out / "results.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "conversation_id", "turn", "transcript", "reply", "decision", "language", "chunk_ids", "tool_calls", "reply_audio", "error"])
        for r in results:
            w.writerow([r["file"], r["conversation_id"] or "", r["turn"], r["transcript"], r["reply"], r["decision"], r["language"], ";".join(r["chunk_ids"]),
                        json.dumps(r["tool_calls"], ensure_ascii=False), r["reply_audio"], r["error"]])
    errs = sum(1 for r in results if r["error"])
    print(f"\nDone: {len(results)} processed, {errs} error(s). Outputs in {out}/" + (f" | Gnani: {voice.stats}" if voice else ""))
    return 1 if errs else 0

if __name__ == "__main__":
    sys.exit(main())
