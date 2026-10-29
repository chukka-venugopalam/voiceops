## WENT_WRONG
- Verified offline recall is below the saved baseline: normal 0.989 to 0.968, ASR 0.989 to 0.968, and conversations 1.000 to 0.944. Reverting the HI_MAP extras and context-query call site did not recover these metrics.
- The latest bounded ASR RAG run completed, but decision accuracy was 94.2%. An earlier ASR RAG run ended in a provider transport error and fell back to offline rules; that earlier result is not valid RAG evidence.
- The synthetic Gnani TTS smoke test produced audio, but the STT smoke test failed. The speech integration status remains incomplete.

## WOULD_FIX
- Investigate generic retrieval ranking and query normalization against the verified offline baselines; do not infer quality from the highly templated held-out set.
- Diagnose the provider-side STT failure and rerun only with an explicitly authorized request budget.
- Review Hindi and Hinglish behavior with native speakers and broaden testing with synthetic, non-sensitive speech.

## BONUS
- Deterministic business rules, section-based knowledge retrieval, optional hybrid-retrieval plumbing, and request-budget ledgers for bounded LLM and embedding use.
- Offline answer accuracy remains 100% for normal/ASR question evaluations; the 23 practice conversations remain 23/23 fully correct.