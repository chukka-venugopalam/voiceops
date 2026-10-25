"""Write replies for the ear (STY-3): month-name dates, order IDs in groups of four, amounts in rupees."""
import re

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_MON = "January February March April May June July August September October November December".split()

def num_words(n: int) -> str:
    if n < 20: return _ONES[n]
    if n < 100: return _TENS[n // 10] + ("-" + _ONES[n % 10] if n % 10 else "")
    if n < 1000: return _ONES[n // 100] + " hundred" + (" " + num_words(n % 100) if n % 100 else "")
    if n < 100000: return num_words(n // 1000) + " thousand" + (" " + num_words(n % 1000) if n % 1000 else "")
    if n < 10000000: return num_words(n // 100000) + " lakh" + (" " + num_words(n % 100000) if n % 100000 else "")
    return str(n)

def _ordinal(d: int) -> str:
    return f"{d}{'th' if 11 <= d % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(d % 10, 'th')}"

def speechify(text: str, lang: str = "en") -> str:
    t = re.sub(r"[*_`#>]+", "", text)
    t = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", "", t)
    # ISO dates -> 27th of October 2026
    t = re.sub(r"\b(20\d\d)-(\d\d)-(\d\d)\b", lambda m: f"the {_ordinal(int(m.group(3)))} of {_MON[int(m.group(2)) - 1]} {m.group(1)}", t)
    # '27 October 2026' -> '27th of October 2026' (English month names only)
    t = re.sub(r"\b(\d{1,2}) (" + "|".join(_MON) + r")( 20\d\d)?\b",
               lambda m: f"{_ordinal(int(m.group(1)))} of {m.group(2)}{m.group(3) or ''}", t)
    # amounts
    def amt(m):
        n = int(m.group(1).replace(",", ""))
        return f"{n} रुपये" if lang == "hi" else f"{num_words(n)} rupee{'s' if n != 1 else ''}"
    t = re.sub(r"(?:₹|Rs\.?|INR)\s*([\d,]+)", amt, t)
    # long IDs -> groups of four
    t = re.sub(r"(?<!\d)(\d{8})(?!\d)", lambda m: f"{m.group(1)[:4]} {m.group(1)[4:]}", t)
    t = re.sub(r"(?<!\d)(\d{12,})(?!\d)", lambda m: " ".join(m.group(1)[i:i + 4] for i in range(0, len(m.group(1)), 4)), t)
    return re.sub(r"\s+", " ", t).strip()
