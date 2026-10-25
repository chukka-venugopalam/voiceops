"""Language detection, spoken-number handling, ID extraction and date parsing (en / hi / hi-en)."""
import re
from datetime import date, timedelta

TODAY = date(2026, 10, 20)
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

_WORD_DIGITS = {}
def _reg(d, *words):
    for w in words: _WORD_DIGITS[w] = d
_reg("0", "zero", "shunya", "sunya", "sifar", "शून्य", "सिफर")
_reg("1", "one", "ek", "एक")
_reg("2", "two", "do", "दो")
_reg("3", "three", "teen", "तीन")
_reg("4", "four", "char", "chaar", "चार")
_reg("5", "five", "panch", "paanch", "पांच", "पाँच")
_reg("6", "six", "chhe", "chhah", "chah", "che", "छह", "छः", "छे")
_reg("7", "seven", "saat", "सात")
_reg("8", "eight", "aath", "aat", "आठ")
_reg("9", "nine", "nau", "नौ")
_MULT = {"double": 2, "triple": 3, "डबल": 2, "ट्रिपल": 3}
_TOKEN = re.compile(r"\d+|[a-z]+|[\u0900-\u097F]+|[^\s\w]", re.I)

def _tokens(text: str):
    return _TOKEN.findall(text.translate(_DEV_DIGITS).lower())

def digit_runs(text: str):
    """Runs of digits, including spoken digits ('one eight five nine', 'double five', 'एक आठ')."""
    runs, cur, mult = [], "", 1
    def flush():
        nonlocal cur
        if cur: runs.append(cur)
        cur = ""
    for t in _tokens(text):
        if t.isdigit():
            cur += t; mult = 1
        elif t in _WORD_DIGITS:
            cur += _WORD_DIGITS[t] * mult; mult = 1
        elif t in _MULT:
            mult = _MULT[t]
        elif t in ("-", ","):
            continue
        else:
            mult = 1; flush()
    flush()
    return runs

def spoken_to_digits(text: str) -> str:
    """Rewrites runs of 2+ spoken digit words as digits ('two zero two six' -> '2026')."""
    # simple, explicit implementation (avoids reconstructing the single original word)
    toks = _tokens(text); i = 0; res = []
    while i < len(toks):
        t = toks[i]; j = i; digits = ""; m = 1; n_words = 0
        while j < len(toks):
            u = toks[j]
            if u in _MULT: m = _MULT[u]; j += 1; continue
            if u in _WORD_DIGITS: digits += _WORD_DIGITS[u] * m; m = 1; n_words += 1; j += 1; continue
            break
        if n_words >= 2: res.append(digits); i = j
        else: res.append(t); i += 1
    return " ".join(res)

_LAST4 = re.compile(r"last\s+(?:four|4)(?:\s+digits?)?|aakhri\s+(?:char|chaar|4)|आख़िरी\s+चार|आखिरी\s+चार|आख़िरी\s*4|आखिरी\s*4", re.I)

_VERIFIER_CONTEXT = re.compile(
    r"last\s+(?:four|4)(?:\s+digits?)?|registered\s+(?:phone|mobile)|"
    r"(?:phone|mobile)\s+(?:number|no\.?|ending)|pinc?ode|pin\s*code|postal\s+code|"
    r"आख़िरी\s*चार|आखिरी\s*चार|आख़िरी\s*4|आखिरी\s*4|"
    r"पंजीकृत\s*(?:फ़ोन|फोन|मोबाइल)|फोन\s*नंबर|फ़ोन\s*नंबर|पिनकोड|पिन\s*कोड|अंक",
    re.I,
)
_SKU_CODE = re.compile(r"\b(?:SKU\s*)?[A-Z]{2,5}\s*[-–]\s*\d{4}\b", re.I)
_CODE_FILLERS = {
    "is", "it", "my", "number", "no", "the", "last", "four", "4", "digit", "digits",
    "hai", "hain", "he", "ji", "please", "yes", "yeah", "yep", "mera", "meri", "ke",
    "aakhri", "akhir", "char", "chaar", "ank", "ankh", "है", "हैं", "जी", "हाँ", "हां",
    "आख़िरी", "आखिरी", "चार", "अंक", "नंबर", "फोन", "फ़ोन", "पिनकोड", "पिन",
}

def _is_bare_verifier_answer(text: str) -> bool:
    """Accept a bare code only when the conversation is already awaiting verification."""
    toks = _tokens(text.translate(_DEV_DIGITS).lower())
    if not toks:
        return False
    has_code = any(t.isdigit() or t in _WORD_DIGITS for t in toks)
    return has_code and all(t.isdigit() or t in _WORD_DIGITS or t in _MULT or t in _CODE_FILLERS or not t.isalnum() for t in toks)

def extract_ids(text: str, expect_verifier: bool = False) -> dict:
    """Extract order IDs and verification values without treating product SKUs as caller IDs.

    A four-digit value is only a verifier when the message identifies it as such, is a
    pincode/phone answer, or the conversation is explicitly waiting for verification.
    """
    order = ver = None
    explicit_verifier = bool(_VERIFIER_CONTEXT.search(text))
    # SKU suffixes (e.g. ELE-0008) are product identifiers, never phone suffixes.
    cleaned = _SKU_CODE.sub(" ", text)
    cleaned = _LAST4.sub(" ", cleaned)  # 'last four' must not be read as a digit
    verifier_context = explicit_verifier or (expect_verifier and _is_bare_verifier_answer(cleaned))
    for r in digit_runs(cleaned):
        if len(r) == 8 and not order:
            order = r
        elif len(r) == 12 and not order:
            order, ver = r[:8], r[8:]
        elif len(r) == 14 and not order:
            order, ver = r[:8], r[8:]
        elif len(r) == 6 and not ver and verifier_context:
            ver = r
        elif len(r) == 4 and r not in ("2025", "2026", "2027") and not ver and verifier_context:
            ver = r
    return {"order_id": order, "verifier": ver}

# ---------------- language ----------------
_SCRIPTS = [("ta", "\u0B80-\u0BFF"), ("te", "\u0C00-\u0C7F"), ("kn", "\u0C80-\u0CFF"), ("ml", "\u0D00-\u0D7F"),
            ("bn", "\u0980-\u09FF"), ("gu", "\u0A80-\u0AFF"), ("pa", "\u0A00-\u0A7F")]
_HINGLISH = set("""hai hain kya mujhe mera meri mere kab kaise kahan kyun nahi nhi chahiye aur ka ki ko mein tak abhi kitne kitna
din paisa paise wapas vapas aaya aaye aayega aayegi kijiye kijiye raha rahi rahe hoon hun bhi toh karna karo karein karun
sakta sakti sakte bataiye batao bataye madad aapka aapki aap hum humne maine isko usko lekin magar abhi tak pehle baad
mil mila milega milegi hua hui hue gaya gayi diya liya kiya lagta lagega bhejna bhejo dobara phir sirf bohot bahut""".split())
_MARATHI = ("आहे", "मला", "काय", "माझा", "माझी", "नाही", "करा")

def detect_language(text: str, prev: str = "en") -> str:
    t = text.strip()
    letters = [c for c in t if c.isalpha()]
    if not letters: return prev
    for code, rng in _SCRIPTS:
        if sum(1 for c in letters if re.match(f"[{rng}]", c)) / len(letters) > 0.3: return code
    dev = sum(1 for c in letters if "\u0900" <= c <= "\u097F")
    if dev / len(letters) > 0.3:
        return "mr" if any(w in t for w in _MARATHI) else "hi"
    words = re.findall(r"[a-z]+", t.lower())
    if sum(1 for w in words if w in _HINGLISH) >= 1: return "hi-en"
    if len(words) <= 2 and prev in ("hi", "hi-en"): return prev
    return "en"

# ---------------- dates ----------------
_MONTHS = {"january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4, "may": 5,
           "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9,
           "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
           "जनवरी": 1, "फरवरी": 2, "फ़रवरी": 2, "मार्च": 3, "अप्रैल": 4, "मई": 5, "जून": 6, "जुलाई": 7, "अगस्त": 8,
           "सितंबर": 9, "सितम्बर": 9, "अक्टूबर": 10, "अक्तूबर": 10, "नवंबर": 11, "नवम्बर": 11, "दिसंबर": 12, "दिसम्बर": 12}
_MN = "|".join(sorted((re.escape(m) for m in _MONTHS), key=len, reverse=True))
_B, _E = r"(?<![\w\u0900-\u097F])", r"(?![\w\u0900-\u097F])"

def parse_dates(text: str, today: date = TODAY):
    """Dates mentioned in text, in order of appearance. Year defaults to 2026."""
    s = spoken_to_digits(text).lower()
    found = []
    def add(pos, d):
        found.append((pos, d))
    for m in re.finditer(r"(20\d\d)-(\d\d)-(\d\d)", s):
        try: add(m.start(), date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError: pass
    for m in re.finditer(_B + r"(\d{1,2})[/-](\d{1,2})[/-](20\d\d)" + _E, s):
        try: add(m.start(), date(int(m.group(3)), int(m.group(2)), int(m.group(1))))
        except ValueError: pass
    for m in re.finditer(_B + r"(\d{1,2})(?:st|nd|rd|th)?\s*(?:of\s+)?(" + _MN + r")" + _E + r"(?:[\s,]+(20\d\d))?", s):
        try: add(m.start(), date(int(m.group(3) or today.year), _MONTHS[m.group(2)], int(m.group(1))))
        except ValueError: pass
    for m in re.finditer(_B + r"(" + _MN + r")\s+(\d{1,2})(?:st|nd|rd|th)?" + _E + r"(?:[\s,]+(20\d\d))?", s):
        try: add(m.start(), date(int(m.group(3) or today.year), _MONTHS[m.group(1)], int(m.group(2))))
        except ValueError: pass
    for pat, off in ((r"day before yesterday|parso|parson|परसों", -2), (r"yesterday|(?<![a-z])kal(?![a-z])|कल", -1),
                     (r"(?<![a-z])today(?![a-z])|(?<![a-z])aaj(?![a-z])|आज", 0)):
        for m in re.finditer(pat, s): add(m.start(), today + timedelta(days=off))
    for m in re.finditer(r"(\d+)\s*(?:days?\s*ago|din\s*pehle|दिन\s*पहले)", s):
        add(m.start(), today - timedelta(days=int(m.group(1))))
    found.sort(key=lambda x: x[0])
    out = []
    for _, d in found:
        if d not in out: out.append(d)
    return out

_METHODS = [(r"net\s*banking|netbanking|नेट\s*बैंकिंग", "netbanking"), (r"(?<![a-z])u\s*p\s*i(?![a-z])|यूपीआई", "upi"),
            (r"cash\s*on\s*delivery|\bcod\b|कैश\s*ऑन\s*डिलीवरी", "cod"),
            (r"credit\s*card|debit\s*card|(?<![a-z])card(?![a-z])|कार्ड", "card"), (r"wallet|वॉलेट", "wallet")]

def detect_payment_method(text: str):
    s = text.lower()
    for pat, name in _METHODS:
        if re.search(pat, s): return name
    return None
