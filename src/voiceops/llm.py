"""Provider-agnostic LLM client (stdlib only). Providers: anthropic, openai, groq, gemini, openrouter,
any OpenAI-compatible endpoint (LLM_BASE_URL), or stub (no network; the agent falls back to rules)."""
import hashlib, json, os, pathlib, re, time, urllib.error, urllib.request

class LLMError(Exception): pass

_DEFAULTS = {   # (base_url, default model). Set LLM_MODEL yourself: model names and free quotas change often.
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    # Gemini 2.5 models were retired in Oct 2026; run `python scripts/preflight.py --models` to see what your key can use.
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.1-flash-lite"),
    "openrouter": ("https://openrouter.ai/api/v1", "openai/gpt-4o-mini"),
}

class BaseLLM:
    name, available, model = "base", False, ""
    def complete(self, system, messages, max_tokens=600, temperature=0.0) -> str:
        raise NotImplementedError

class StubLLM(BaseLLM):
    name = "stub"
    def complete(self, system, messages, max_tokens=600, temperature=0.0):
        raise LLMError("stub LLM: no model configured")

class _HTTPLLM(BaseLLM):
    available = True
    def __init__(self, model, api_key, base_url, cache=True, timeout=60):
        self.model, self.api_key, self.base_url, self.timeout = model, api_key, base_url.rstrip("/"), timeout
        self.cache_dir = pathlib.Path(os.environ.get("LLM_CACHE_DIR", ".cache/llm")) if cache and os.environ.get("LLM_CACHE", "1") != "0" else None
        self.calls = 0; self.cache_hits = 0
        self.min_interval = float(os.environ.get("LLM_MIN_INTERVAL") or 0)      # seconds between real calls (free tiers: e.g. 4.5 = 13 calls/minute)
        self.retries = int(os.environ.get("LLM_RETRIES") or 4); self._last = 0.0
    def _key(self, system, messages, max_tokens, temperature):
        raw = json.dumps([self.name, self.model, system, messages, max_tokens, temperature], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    @staticmethod
    def _retry_delay(body: str):
        m = re.search(r'retryDelay"?\s*[:=]\s*"?(\d+(?:\.\d+)?)s', body) or re.search(r"retry in (\d+(?:\.\d+)?)s", body, re.I)
        return float(m.group(1)) if m else None
    def _post(self, url, headers, payload, method="POST"):
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        last = None
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json", **headers})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")
                last = LLMError(f"HTTP {e.code}: {body[:400]}")
                if e.code == 429:
                    delay = self._retry_delay(body)
                    if "PerDay" in body or (delay and delay > 120):
                        raise LLMError("DAILY QUOTA EXHAUSTED for this model/key (use another model, another key, or enable billing): " + body[:300])
                    time.sleep(min((delay or 5) + 1, 70)); continue
                if e.code in (500, 502, 503, 504, 529): time.sleep(1.5 ** attempt + 0.5); continue
                raise last
            except (urllib.error.URLError, TimeoutError) as e:
                last = LLMError(f"network error: {e}"); time.sleep(1.5 ** attempt + 0.5)
        raise last or LLMError("request failed")
    def list_models(self):
        """Model ids this key can use (OpenAI-compatible /models, or Anthropic /v1/models)."""
        hdr = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"} if self.name == "anthropic" else {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}/v1/models" if self.name == "anthropic" else f"{self.base_url}/models"
        r = self._post(url, hdr, None, method="GET")
        return sorted((m.get("id") or m.get("name") or "").replace("models/", "") for m in r.get("data", r.get("models", [])))
    def complete(self, system, messages, max_tokens=600, temperature=0.0):
        key = self._key(system, messages, max_tokens, temperature)
        if self.cache_dir:
            f = self.cache_dir / f"{key}.txt"
            if f.exists(): self.cache_hits += 1; return f.read_text(encoding="utf-8")
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0: time.sleep(wait)
        out = self._call(system, messages, max_tokens, temperature)
        self._last = time.time(); self.calls += 1
        if self.cache_dir and out:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / f"{key}.txt").write_text(out, encoding="utf-8")
        return out
    def _call(self, system, messages, max_tokens, temperature): raise NotImplementedError

class AnthropicLLM(_HTTPLLM):
    name = "anthropic"
    def _call(self, system, messages, max_tokens, temperature):
        r = self._post(f"{self.base_url}/v1/messages", {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
                       {"model": self.model, "max_tokens": max_tokens, "temperature": temperature, "system": system, "messages": messages})
        return "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text").strip()

class OpenAICompatLLM(_HTTPLLM):
    name = "openai-compat"
    def _call(self, system, messages, max_tokens, temperature):
        body = {"model": self.model, "messages": [{"role": "system", "content": system}] + messages,
                "temperature": temperature, "max_tokens": max_tokens}
        url, hdr = f"{self.base_url}/chat/completions", {"Authorization": f"Bearer {self.api_key}"}
        try:
            r = self._post(url, hdr, body)
        except LLMError as e:       # newer OpenAI models want max_completion_tokens and no temperature
            if "max_tokens" in str(e) or "temperature" in str(e):
                body.pop("max_tokens"); body.pop("temperature"); body["max_completion_tokens"] = max_tokens
                r = self._post(url, hdr, body)
            else: raise
        return (r["choices"][0]["message"].get("content") or "").strip()

def get_llm(provider=None) -> BaseLLM:
    provider = (provider or os.environ.get("LLM_PROVIDER") or "").lower()
    key = os.environ.get("LLM_API_KEY", "")
    if not provider:   # infer from conventional key variables
        for env, prov in (("ANTHROPIC_API_KEY", "anthropic"), ("OPENAI_API_KEY", "openai"), ("GROQ_API_KEY", "groq"),
                          ("GEMINI_API_KEY", "gemini"), ("GOOGLE_API_KEY", "gemini"), ("OPENROUTER_API_KEY", "openrouter")):
            if os.environ.get(env): provider, key = prov, key or os.environ[env]; break
        else:
            provider = "anthropic" if key and not os.environ.get("LLM_BASE_URL") else ("openai_compat" if key else "stub")
    if provider in ("stub", "none", "offline"): return StubLLM()
    if not key:
        key = os.environ.get({"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "groq": "GROQ_API_KEY",
                              "gemini": "GEMINI_API_KEY", "openrouter": "OPENROUTER_API_KEY"}.get(provider, "LLM_API_KEY"), "")
    if not key: return StubLLM()
    model = os.environ.get("LLM_MODEL", "")
    if provider == "anthropic":
        return AnthropicLLM(model or "claude-haiku-5-5", key, os.environ.get("LLM_BASE_URL", "https://api.anthropic.com"))
    base, dmodel = _DEFAULTS.get(provider, (os.environ.get("LLM_BASE_URL", ""), ""))
    base = os.environ.get("LLM_BASE_URL", base)
    if not base: raise LLMError("Set LLM_BASE_URL for an OpenAI-compatible provider")
    llm = OpenAICompatLLM(model or dmodel, key, base)
    llm.name = provider
    return llm

def parse_json(text: str):
    """Extract the first JSON object from model output (tolerates code fences / chatter)."""
    if not text: return None
    s = text.strip().replace("```json", "```")
    if "```" in s:
        parts = [p for p in s.split("```") if "{" in p]
        s = parts[0] if parts else s
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b <= a: return None
    try: return json.loads(s[a:b + 1])
    except Exception: return None
