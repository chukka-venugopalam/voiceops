"""Section-level chunking (keeps [SECTION-ID], status, effective_from) + BM25 retrieval baseline."""
import csv, glob, math, os, re, unicodedata
from collections import Counter
from .config import dataset_dir
from .facts import find_products, load_products, _normalize_spoken_numbers
from .textnorm import parse_dates
from .embed import rrf_fuse

HEAD = re.compile(r"^## \[([^\]]+)\]\s*(.*)$", re.M)
DOC_WEIGHT = {"policy": 1.0, "agent_guide": 0.95, "help": 0.9, "meta": 0.8, "faq": 0.7, "marketing": 0.45, "catalog": 0.9}
TOKEN = re.compile(r"[\w\u0900-\u097F]+")      # keeps Devanagari matras inside words

# tiny Hindi/Hinglish -> English keyword map (stopgap; replace with an LLM query rewrite)
HI_MAP = {
    "wapas": "return", "vapas": "return", "लौटा": "return", "वापस": "return", "रिटर्न": "return",
    "paisa": "refund", "रिफ़ंड": "refund", "रिफंड": "refund", "पैसे": "refund",
    "kitne din": "days window", "कितने दिन": "days window", "kab": "when timeline", "कब": "when timeline",
    "कैंसिल": "cancel", "रद्द": "cancel", "ऑर्डर": "order", "शिप": "shipped",
    "वारंटी": "warranty", "खराब": "damaged", "टूटा": "damaged", "गलत": "wrong", "पिकअप": "pickup",
    "एड्रेस": "address", "पता": "address", "फीस": "fee", "फ़ीस": "fee", "मुफ़्त": "free", "मुफ्त": "free",
    "कैश ऑन डिलीवरी": "cash on delivery cod", "वॉलेट": "wallet", "बैंक": "bank", "बदल": "change",
    "kaam nahi": "not working troubleshooting", "kharab": "damaged", "toota": "damaged", "galat": "wrong",

}

STOP = set("""a an the is are was were be been am do does did can could i me my mine you your we our it its of to in on at for with
and or if as by from this that these those what which who how so then than there about please hi hello yes no not
kya hai hain ka ki ke ko pe par se mein me aur ya to toh kaise kab tak aa aaya aaye mera meri mere mujhe main hum
kar kare karna karun kr sakta sakti sakte chahiye hoga hogi tha thi the nahi na ek bhi abhi ab so
है हैं का की के को पे पर से में मैं मेरा मेरी मुझे और या क्या कैसे कब तक कर सकता सकती सकते हूँ हो था थी नहीं भी अब""".split())

def tokenize(text: str, drop_stop: bool = True):
    toks = TOKEN.findall(text.lower())
    return [t for t in toks if t not in STOP] if drop_stop else toks

def _frontmatter(txt):
    m = re.match(r"^---\n(.*?)\n---\n", txt, re.S)
    meta = {}
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1); meta[k.strip()] = v.strip()
        txt = txt[m.end():]
    return meta, txt

def load_chunks(root=None):
    root = root or dataset_dir()
    chunks = []
    for path in sorted(glob.glob(os.path.join(root, "kb", "*.md"))):
        txt = open(path, encoding="utf-8").read().replace("\r\n", "\n")
        meta, body = _frontmatter(txt)
        parts = list(HEAD.finditer(body))
        for i, m in enumerate(parts):
            end = parts[i + 1].start() if i + 1 < len(parts) else len(body)
            chunks.append(dict(id=m.group(1), title=m.group(2).strip(), text=body[m.end():end].strip(),
                               doc_id=meta.get("doc_id"), doc_type=meta.get("doc_type"), status=meta.get("status"),
                               effective_from=meta.get("effective_from"), language=meta.get("language", "en"),
                               source=os.path.basename(path)))
    with open(os.path.join(root, "data", "products.csv"), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            t = (f"{r['title']} {r['brand']} {r['sku']} category {r['return_category']} price {r['price_inr']} "
                 f"fulfilled by {r['fulfilled_by']} assured {r['assured']} warranty {r['warranty_months']} months {r['description']}")
            chunks.append(dict(id=f"products.csv:{r['sku']}", title=r["title"], text=t, doc_id="catalog", doc_type="catalog",
                               status="current", effective_from=None, language="en", source="products.csv"))
    return chunks

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.toks = [tokenize(d) for d in docs]
        self.N = len(docs)
        self.avg = sum(len(t) for t in self.toks) / max(1, self.N)
        self.tf = [Counter(t) for t in self.toks]
        self.df = Counter()
        for t in self.toks: self.df.update(set(t))
    def scores(self, query):
        q = tokenize(query); out = []
        for i, tf in enumerate(self.tf):
            dl, s = len(self.toks[i]), 0.0
            for w in q:
                if w in tf:
                    idf = math.log(1 + (self.N - self.df[w] + 0.5) / (self.df[w] + 0.5))
                    s += idf * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * dl / self.avg))
            out.append(s)
        return out

def expand_query(q: str) -> str:
    q = _normalize_spoken_numbers(re.sub(r"\bre\s+fund\b", "refund", q, flags=re.I))
    low, extra = unicodedata.normalize("NFD", q).lower(), []
    for k, v in HI_MAP.items():
        if unicodedata.normalize("NFD", k) in low: extra.append(v)
    return q + " " + " ".join(extra)

TIME_WORDS = re.compile(r"refund|days?|date|when|how long|how soon|how fast|timeline|business|kab|tak|कब|कितने|दिन|तारीख", re.I)

_ID_LIKE = re.compile(r"\b[A-Za-z]{2,5}[-_ ]?\d{2,}\b|\d+")
_NONTOPIC = set("last four digits digit pincode pin phone number mobile order id yes yeah okay ok sure thanks thank correct right hello hi".split())

def content_tokens(text):
    """Topic words of a customer message: IDs, SKUs, digit strings and bare confirmations do not count."""
    return [w for w in tokenize(_ID_LIKE.sub(" ", text)) if w not in _NONTOPIC and len(w) > 2]

def context_query(history, n=1):
    """Latest n customer turn(s) that carry a topic, oldest first. A follow-up such as 'and for COD?' needs the earlier
    question, and the immediately previous turn is often just an order ID or the 4 verification digits."""
    out = []
    for role, msg in reversed(history):
        if role == "customer" and content_tokens(msg):
            out.append(msg)
            if len(out) == n: break
    return " ".join(reversed(out))[:300]

class Retriever:
    """Policy and catalog are ranked separately so product rows cannot crowd out policy sections."""
    def __init__(self, root=None, use_embed=False):
        self.root = root or dataset_dir()
        self.chunks = load_chunks(self.root)
        self.by_id = {c["id"]: c for c in self.chunks}
        self.products = load_products(self.root)
        self.index = BM25([c["title"] + " " + c["title"] + " " + c["text"] for c in self.chunks])
        self.embed, self.embed_errors = None, 0
        if use_embed:
            try:
                from .embed import EmbedIndex
                pol = [c for c in self.chunks if c["doc_type"] != "catalog"]
                self.embed = EmbedIndex.from_env([c["id"] for c in pol], [c["title"] + ". " + c["text"] for c in pol])
            except Exception:
                self.embed = None

    EMBED_TOP = 12

    def _fuse(self, out, query):
        """Fuse policy rankings only; existing evidence-preference and catalog logic remains authoritative."""
        if not self.embed: return out
        try:
            hits = self.embed.rank(query, top=self.EMBED_TOP)
        except Exception:
            self.embed_errors += 1
            if self.embed_errors >= 3: self.embed = None
            return out
        pol = [(score, chunk) for score, chunk in out if chunk["doc_type"] != "catalog"]
        cat = [(score, chunk) for score, chunk in out if chunk["doc_type"] == "catalog"]
        bm = [chunk["id"] for score, chunk in pol if score > 0][:30]
        em = [cid for cid, _ in hits]
        fused = rrf_fuse([bm, em])
        pol = sorted(((fused.get(chunk["id"], 0.0) * 100, chunk) for _, chunk in pol), key=lambda x: -x[0])
        return sorted(pol + cat, key=lambda x: -x[0])
    def _ranked(self, query):
        raw = self.index.scores(expand_query(query))
        out = []
        dates = parse_dates(query)
        current_returns = self.by_id.get("RET-3", {}).get("effective_from")
        historical_return = bool(current_returns and any(
            d.isoformat() < current_returns for d in dates
        ))
        for c, s_ in zip(self.chunks, raw):
            superseded_weight = 0.5 if c["status"] == "superseded" else 1.0
            if c["id"] == "RETv3-2" and historical_return:
                superseded_weight = 1.0
            w = DOC_WEIGHT.get(c["doc_type"], 0.8) * superseded_weight
            out.append((s_ * w, c))
        out.sort(key=lambda x: -x[0])
        return self._fuse(out, query)

    def _evidence_ids(self, query, products):
        low = expand_query(query).lower()
        ids = []
        categories = {p["return_category"] for p in products}

        def add(*values):
            for value in values:
                if value not in ids:
                    ids.append(value)

        if re.search(r"phone number|email|address|details of (?:the )?(?:person|customer|account)", low):
            add("IDV-4", "ESC-4")
        elif re.search(r"never (?:placed|ordered|made)|did not (?:place|order)|order .* never", low):
            add("ESC-1")

        if re.search(r"refund|re fund|cash on delivery|\bcod\b|upi|net banking|credit card", low):
            if re.search(r"cash on delivery|\bcod\b", low):
                if re.search(r"refund|money back", low):
                    add("REF-3", "REF-2", "FAQH-2")
                else:
                    add("PAY-2")
            elif re.search(r"return(?:ed)?|pickup|quality check|qc", low) and re.search(r"status|initiated|started|refund", low):
                add("RET-7", "REF-1", "REF-2")
            else:
                add("REF-2")
                if parse_dates(query) and re.search(r"started|initiated|shuru|kab|कब|when|date", low):
                    add("KB-00.3")
                elif re.search(r"wallet|how fast|how soon|how long|timeline|kab|कब|दिन", low):
                    add("KB-00.1")
                elif re.search(r"when|date|started|initiated", low):
                    add("KB-00.3")
            if re.search(r"\bplus\b", low):
                add("PLS-2")

        if re.search(r"return|send(?: it)? back|return window|लौटा|वापस|रिटर्न", low):
            if any(d.isoformat() < "2026-08-01" for d in parse_dates(query)) and "electronics" in categories:
                add("RET-3", "RETv3-2")
            else:
                add("RET-2")
                if "electronics" in categories:
                    add("RET-3")
            if "beauty_sealed" in categories or re.search(r"opened|open(ed)?|tried|used|खोल", low):
                add("RET-4")
            if "innerwear" in categories:
                add("RET-5")
            if "fashion" in categories:
                add("RET-8")
            if re.search(r"fee|charge|collection|pickup|pick-up|free", low):
                add("RET-6")

        if re.search(r"warranty|assured|claim|वारंटी", low):
            add("WAR-1", "WAR-2")

        damaged = re.search(r"damag|defect|torn|crack|broken|missing|never received|not received|wrong item|खराब|टूटा", low)
        if damaged:
            if "fashion" in categories and re.search(r"fee|charge|collection|pickup|free|torn", low):
                add("DWM-2", "RET-6", "DWM-1")
            elif re.search(r"never received|not received|missing|delivered but", low):
                add("TRK-2", "DWM-1")
            elif re.search(r"three days|3 days|after 48 hours|more than 48 hours", low):
                add("DWM-4", "DWM-1")
            elif any(float(p.get("price_inr") or 0) > 10000 for p in products):
                add("DWM-3", "ESC-1")
            else:
                add("DWM-1", "DWM-2")
            if "fashion" in categories and re.search(r"fee|charge|collection|pickup|free|torn", low):
                add("RET-6", "DWM-2")

        if re.search(r"refuse to collect|refused to collect|pickup at the door|collect my return", low):
            add("RVP-3", "RET-4")
        if re.search(r"\bplus\b", low) and re.search(r"pickup fee|pickup fees|return fee|fee on return", low):
            add("PLS-2", "FAQH-4")
        if re.search(r"already on its way|already shipped|cannot cancel|stop it", low):
            add("CAN-2")

        return ids

    def search(self, query: str, k: int = 4, catalog_k: int = 1, attach_meta: bool = True):
        normalized_query = expand_query(query)
        products = find_products(normalized_query, self.products, limit=2)
        if not products:
            query_tokens = tokenize(normalized_query)
            for sku, product in self.products.items():
                title_tokens = tokenize(product["title"].split(",", 1)[0])
                if title_tokens and any(
                    query_tokens[start:start + len(title_tokens)] == title_tokens
                    for start in range(len(query_tokens) - len(title_tokens) + 1)
                ):
                    products.append(product)
                    if len(products) == 2:
                        break
        preferred_ids = self._evidence_ids(normalized_query, products)
        ranked = self._ranked(query)
        ranked_by_id = {c["id"]: (s_, c) for s_, c in ranked}
        preferred = [ranked_by_id[cid] for cid in preferred_ids if cid in ranked_by_id]
        preferred_ids_present = {c["id"] for _, c in preferred}
        kb = (preferred + [
            (s_, c) for s_, c in ranked
            if c["doc_type"] != "catalog" and s_ > 0 and c["id"] not in preferred_ids_present
        ])[:k]
        matched_catalog_ids = [f"products.csv:{p['sku']}" for p in products]
        catalog_ranked = [
            (s_, c) for s_, c in ranked
            if c["doc_type"] == "catalog" and s_ > 0
        ]
        matched_catalog = [ranked_by_id[cid] for cid in matched_catalog_ids if cid in ranked_by_id] if catalog_k else []
        matched_catalog.sort(key=lambda item: -item[0])
        catalog_ids = {c["id"] for _, c in matched_catalog}
        cat = []
        if catalog_k:
            cat = matched_catalog + [
                item for item in catalog_ranked if item[1]["id"] not in catalog_ids
            ][:max(catalog_k, len(matched_catalog))]
        if matched_catalog:
            insertion = min(len(preferred), 2, len(kb))
            combined = kb[:insertion] + cat + kb[insertion:]
        else:
            combined = kb + cat
        res = [dict(score=round(s_, 3), attached=False, **c) for s_, c in combined]
        if attach_meta and TIME_WORDS.search(query):          # precedence + business-day rules (KB-00.1 / KB-00.3)
            have = {r["id"] for r in res}
            for cid in ("KB-00.1", "KB-00.3"):
                if cid in self.by_id and cid not in have:
                    res.append(dict(score=0.0, attached=True, **self.by_id[cid]))
        return res
