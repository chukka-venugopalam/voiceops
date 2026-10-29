"""Optional hybrid retrieval embeddings with an enforced persistent 60-request cap.
No calls happen unless explicitly used by a retrieval evaluator or VOICEOPS_ENABLE_HYBRID_RAG=1.
Every HTTP attempt, including retries, consumes one budget unit before it is sent.
"""
import hashlib, json, math, os, pathlib, time, urllib.error, urllib.request

_KEY_ENV = {"openai": "OPENAI_API_KEY", "groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
MAX_REQUESTS = 60
BATCH_SIZE = 32
ROOT = pathlib.Path(__file__).resolve().parents[2]
BUDGET_FILE = ROOT / ".cache" / "embed" / "voiceops_v9_1_2_request_budget.json"

class RequestBudgetExceeded(RuntimeError):
    pass

class RequestBudget:
    """Persistent, fail-closed cap across separate processes; do not run embedding jobs concurrently."""
    def __init__(self, path=None, limit=MAX_REQUESTS):
        self.path = pathlib.Path(path) if path is not None else BUDGET_FILE
        self.limit = int(limit)
    def _read(self):
        if not self.path.exists():
            return {"used": 0, "limit": self.limit}
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            used, limit = d["used"], d["limit"]
            if not isinstance(used, int) or used < 0 or limit != self.limit or used > self.limit:
                raise ValueError("invalid budget fields")
            return {"used": used, "limit": limit}
        except Exception as e:
            raise RequestBudgetExceeded("embedding request-budget ledger unreadable; refusing network calls") from e
    def used(self):
        return self._read()["used"]
    def remaining(self):
        return self.limit - self.used()
    def ensure_capacity(self, minimum_calls):
        state = self._read()
        if state["used"] + int(minimum_calls) > self.limit:
            raise RequestBudgetExceeded(
                f"embedding request budget insufficient: used={state['used']}, minimum_needed={minimum_calls}, limit={self.limit}; no requests started")
    def consume(self):
        state = self._read()
        if state["used"] >= self.limit:
            raise RequestBudgetExceeded(f"embedding request budget exhausted at {self.limit}; no further request sent")
        state["used"] += 1
        state["updated_epoch"] = int(time.time())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + f".{os.getpid()}.tmp")
        try:
            temp.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
            os.replace(temp, self.path)
        finally:
            try: temp.unlink()
            except FileNotFoundError: pass
        return state["used"]

def rrf_fuse(rank_lists, k=60):
    out = {}
    for lst in rank_lists:
        for r, cid in enumerate(lst):
            out[cid] = out.get(cid, 0.0) + 1.0 / (k + r + 1)
    return out

def _unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]

def _read_json(path):
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        # Read-only on cache parse failure: --estimate-only must not mutate any file.
        return {}

def _atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, path)
    finally:
        try: temp.unlink()
        except FileNotFoundError: pass

class EmbedIndex:
    last_error = ""
    def __init__(self, ids, texts, model, key, base, cache_dir, query_texts=None, budget=None):
        self.ids, self.model, self.key, self.base = list(ids), model, key, base.rstrip("/")
        self.dir = pathlib.Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        # Separate cache namespace by model AND endpoint (never by API key).
        endpoint_tag = hashlib.sha1((model + "\n" + self.base).encode("utf-8")).hexdigest()[:12]
        self._dfile, self._qfile = self.dir / f"docs_{endpoint_tag}.json", self.dir / f"queries_{endpoint_tag}.json"
        self._dcache, self._qcache = _read_json(self._dfile), _read_json(self._qfile)
        self.budget = budget or RequestBudget()
        queries = list(query_texts or [])
        # Preflight lower-bound budget for the complete operation before any network call.
        doc_missing = self._missing(texts, self._dcache)
        query_missing = self._missing(queries, self._qcache)
        minimum = math.ceil(len(doc_missing) / BATCH_SIZE) + math.ceil(len(query_missing) / BATCH_SIZE)
        self.budget.ensure_capacity(minimum)
        self.vecs = self._embed_docs(texts)
        self.prime_queries(queries)

    def _h(self, text):
        return hashlib.sha1((self.model + "\n" + text).encode("utf-8")).hexdigest()

    def _missing(self, texts, cache):
        return list(dict((self._h(t), t) for t in texts if self._h(t) not in cache).items())

    @staticmethod
    def _settings():
        model = os.environ.get("EMBED_MODEL", "").strip()
        if not model: return None
        try:
            from .llm import _DEFAULTS
            provider = (os.environ.get("EMBED_PROVIDER") or os.environ.get("LLM_PROVIDER") or "").lower()
            base = os.environ.get("EMBED_BASE_URL") or _DEFAULTS.get(provider, ("", ""))[0]
            key = (os.environ.get("EMBED_API_KEY") or os.environ.get("LLM_API_KEY")
                   or os.environ.get(_KEY_ENV.get(provider, "LLM_API_KEY"), ""))
            if not base or not key:
                return {"error": "embedding endpoint/key configuration missing", "model": model, "provider": provider}
            return {"model": model, "provider": provider, "base": base.rstrip("/"), "key": key}
        except Exception:
            return {"error": "embedding provider configuration unavailable", "model": model}

    @classmethod
    def estimate_from_env(cls, ids, texts, query_texts, cache_dir=None, budget=None):
        """Estimate minimum uncached HTTP attempts without making any request or printing secret values."""
        s = cls._settings()
        if not s or s.get("error"):
            return {"configured": bool(s), "available": False, "reason": (s or {}).get("error", "EMBED_MODEL not configured")}
        dir_ = pathlib.Path(cache_dir or os.environ.get("EMBED_CACHE_DIR", ".cache/embed"))
        endpoint_tag = hashlib.sha1((s["model"] + "\n" + s["base"]).encode("utf-8")).hexdigest()[:12]
        dc, qc = _read_json(dir_ / f"docs_{endpoint_tag}.json"), _read_json(dir_ / f"queries_{endpoint_tag}.json")
        def h(t): return hashlib.sha1((s["model"] + "\n" + t).encode("utf-8")).hexdigest()
        dm = len({h(t) for t in texts if h(t) not in dc})
        qm = len({h(t) for t in query_texts if h(t) not in qc})
        b = budget or RequestBudget()
        used = b.used(); rem = b.remaining()
        minimum = math.ceil(dm / BATCH_SIZE) + math.ceil(qm / BATCH_SIZE)
        return {"configured": True, "available": True, "model_set": True, "uncached_documents": dm,
                "uncached_queries": qm, "minimum_requests": minimum, "request_budget_used": used,
                "request_budget_remaining": rem, "hard_limit": b.limit,
                "minimum_fits_with_five_request_retry_margin": used + minimum + 5 <= b.limit}

    def _post(self, texts):
        if not texts: return []
        body = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(self.base + "/embeddings", data=body, method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.key})
        delay = 1.0
        for attempt in range(4):
            # Counts the actual attempt before the request, including every retry.
            self.budget.consume()
            try:
                with urllib.request.urlopen(req, timeout=45) as response:
                    data = json.loads(response.read().decode("utf-8"))
                rows = sorted(data["data"], key=lambda d: d.get("index", 0))
                if len(rows) != len(texts):
                    raise RuntimeError("embedding response count mismatch")
                return [_unit(d["embedding"]) for d in rows]
            except urllib.error.HTTPError as e:
                if e.code in (408, 429, 500, 502, 503, 504) and attempt < 3:
                    time.sleep(delay); delay = min(delay * 2, 4.0); continue
                raise RuntimeError(f"embedding request failed with HTTP status {e.code}; response body suppressed") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < 3:
                    time.sleep(delay); delay = min(delay * 2, 4.0); continue
                raise RuntimeError("embedding network request failed; detail suppressed") from None
            except RequestBudgetExceeded:
                raise
            except Exception as e:
                raise RuntimeError("embedding response invalid; response detail suppressed") from None
        raise RuntimeError("embedding retries exhausted")

    def _embed_docs(self, texts):
        keys = [self._h(t) for t in texts]
        todo = self._missing(texts, self._dcache)
        for i in range(0, len(todo), BATCH_SIZE):
            batch = todo[i:i + BATCH_SIZE]
            vecs = self._post([t[:2000] for _, t in batch])
            for (k, _), v in zip(batch, vecs):
                self._dcache[k] = [round(x, 6) for x in v]
            _atomic_json(self._dfile, self._dcache)
        return [self._dcache[k] for k in keys]

    def prime_queries(self, queries):
        """Embed an iterable of queries in batches so held-out evaluation uses few request units."""
        queries = list(dict.fromkeys(q for q in queries if isinstance(q, str) and q.strip()))
        todo = self._missing(queries, self._qcache)
        for i in range(0, len(todo), BATCH_SIZE):
            batch = todo[i:i + BATCH_SIZE]
            vecs = self._post([t[:2000] for _, t in batch])
            for (k, _), v in zip(batch, vecs):
                self._qcache[k] = [round(x, 6) for x in v]
            _atomic_json(self._qfile, self._qcache)
        return len(todo)

    def rank(self, query, top=12):
        key = self._h(query)
        if key not in self._qcache:
            self.prime_queries([query])
        qv = self._qcache[key]
        sims = []
        for cid, v in zip(self.ids, self.vecs):
            if len(qv) != len(v):
                raise RuntimeError("embedding vector dimension mismatch")
            sims.append((cid, sum(a * b for a, b in zip(qv, v))))
        sims.sort(key=lambda x: -x[1])
        return sims[:top]

    @classmethod
    def from_env(cls, ids, texts, query_texts=None):
        s = cls._settings()
        if not s:
            cls.last_error = "EMBED_MODEL not configured"
            return None
        if s.get("error"):
            cls.last_error = s["error"]
            return None
        try:
            return cls(ids, texts, s["model"], s["key"], s["base"],
                       os.environ.get("EMBED_CACHE_DIR", ".cache/embed"), query_texts=query_texts)
        except Exception as e:
            cls.last_error = str(e)[:200]
            return None
