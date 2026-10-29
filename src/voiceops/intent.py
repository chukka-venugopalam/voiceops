"""Rule-based intent flags. Used as the offline fallback and as a safety net for hard escalation triggers.
The LLM analysis (when available) refines topic / needs_order / search_query; hard triggers are OR-ed."""
import re, unicodedata
from .textnorm import parse_dates, detect_payment_method, digit_runs

def _nfd(x): return unicodedata.normalize("NFD", x)
def _rx(*pats): return re.compile(_nfd("|".join(pats)), re.I)
_NB, _NE = r"(?<![\w\u0900-\u097F])", r"(?![\w\u0900-\u097F])"

HUMAN = _rx(r"\bhuman\b", r"real person", r"speak (?:to|with) (?:a |an |some)\w*", r"talk (?:to|with) (?:a |an |some)\w*",
            r"live agent", r"\bmanager\b", r"\bsupervisor\b", r"customer care executive", r"insaan", r"insan se", r"kisi se baat",
            r"इंसान", r"किसी से बात", r"मैनेजर", r"सुपरवाइज़र", r"असली व्यक्ति", r"व्यक्ति से बात")
LEGAL = _rx(r"consumer court", r"consumer forum", r"legal action", r"legal notice", r"\blawyer\b", r"\badvocate\b", r"\bsue\b",
            _NB + r"court" + _NE, r"police", r"\bfir\b", r"वकील", r"कोर्ट", r"अदालत", r"पुलिस", r"कानूनी", r"उपभोक्ता")
SAFETY = _rx(r"swell", r"swollen", r"bulg", r"burning smell", r"burnt smell", r"smell\w* (?:of )?burn", r"spark", r"smoke",
             r"caught fire", r"on fire", r"exploded", r"explod", r"electric shock", r"injur", r"aag lag", r"आग", r"चिंगारी",
             r"धुआँ", r"धुआं", r"सूज", r"फूल (?:गई|गया)", r"जलने", r"चोट")
UNRECOGNIZED = _rx(r"never placed", r"didn'?t place", r"did not place", r"not placed by me", r"never ordered", r"didn'?t order",
                   r"did not order", r"unauthori[sz]ed", r"\bfraud", r"hacked", r"not made by me", r"i didn'?t make",
                   r"order nahi (?:kiya|diya)", r"maine nahi (?:kiya|diya)", r"maine .{0,20}nahi (?:kiya|diya)",
                   r"नहीं (?:किया|दिया)", r"धोखाधड़ी", r"बिना मेरे")
COMPENSATION = _rx(r"compensat", r"voucher", r"harjana", r"हर्जाना", r"मुआवज", r"वाउचर", r"goodwill credit", r"extra credit")
CANCEL = _rx(r"cancel", r"कैंसिल", r"रद्द", r"radd", r"\bstop (?:it|the order|my order|this order|shipment)\b", r"stop it", r"order ko rok", r"ऑर्डर रोक", r"इसे रोक")

TOPICS = [  # order matters: first match wins
    ("damage", _rx(r"damag", r"broken", r"\btorn\b", r"\bcrack(?:ed|ing)?\b", r"defect", r"wrong item", r"wrong product", r"missing item", r"टूट", r"खराब", r"गलत सामान", r"galat saman", r"toota", r"kharab", r"swell", r"burn")),
    ("cancel", CANCEL),
    ("refund", _rx(r"re[\s-]*fund", r"रिफ़ंड", r"रिफंड", r"money back", r"paise wapas", r"पैसे वापस")),
    ("return", _rx(r"\breturn", r"exchange", r"send back", r"send\s+(?:it|this|them)\s+back", r"doesn[’']?t fit", r"does not fit", r"replace", r"wapas kar", r"लौटा", r"वापस कर", r"रिटर्न", r"एक्सचेंज")),
    ("delivery", _rx(r"deliver", r"\bship\b", r"not at home", r"not home", r"ghar (?:pe|par) nahi", r"घर पर नहीं", r"arriv", r"\blate\b", r"where is", r"track", r"shipp", r"parcel", r"package", r"kab tak aa", r"कब आ", r"पहुं", r"डिलीवरी", r"शिप", r"aaya nahi", r"nahi aaya")),
    ("warranty", _rx(r"warrant", r"वारंटी", r"assured", r"guarantee", r"claim")),
    ("payment", _rx(r"\bcod\b", r"cash on delivery", r"\bupi\b", r"payment", r"wallet", r"वॉलेट", r"कैश ऑन", r"pay ")),
    ("coupon", _rx(r"coupon", r"promo", r"कूपन")),
    ("plus", _rx(r"\bplus\b", r"membership", r"सदस्य")),
    ("troubleshoot", _rx(r"not working", r"not charging", r"bluetooth", r"\bsync", r"power on", r"turn on", r"earbud", r"काम नहीं", r"चार्ज", r"kaam nahi")),
]
ORDER_SPECIFIC = _rx(r"my order", r"my refund", r"my package", r"my parcel", r"my delivery", r"my item", r"mera order", r"meri order",
                     r"mere order", r"mera refund", r"mera parcel", r"mera saman", r"मेरा ऑर्डर", r"मेरी ऑर्डर", r"मेरा रिफ़ंड",
                     r"मेरा रिफंड", r"मेरी डिलीवरी", r"where is my", r"kahan hai", r"कहाँ है", r"कहां है", r"hasn'?t arrived", r"has not arrived",
                     r"not arrived", r"still not", r"abhi tak nahi", r"अभी तक नहीं", r"cancel it", r"cancel my", r"order cancel",
                     r"ऑर्डर कैंसिल", r"cancel karna", r"refund status", r"status of my", r"track my", r"order (?:ke baare|ki|ka)", r"ऑर्डर के बारे")
MENTIONS_ORDER = _rx(r"\border\b", r"ऑर्डर", r"parcel", r"package", r"shipment", r"delivery")
THIRD_PARTY = _rx(r"(?:phone|number|address|details|name|email).{0,30}\b(?:of|about)\b.{0,20}(?:person|someone|neighbou?r|customer|who ordered|another)",
                  r"person who ordered", r"someone else'?s (?:order|details|address|phone)", r"dusre (?:ka|ki|ke)", r"किसी और (?:का|की|के)", r"पड़ोसी")
INLINE_STATUS = _rx(r"already (?:on its way|shipped|dispatched)", r"has (?:already )?shipped", r"out for delivery", r"on its way", r"रास्ते में", r"शिप हो चुका", r"शिप हो गया", r"ऑर्डर शिप हो गया", r"nikal chuka", r"ship ho chuka", r"ship ho gaya", r"order ship ho gaya")
THIRD_TIME = _rx(r"third time", r"3rd time", r"teesri baar", r"तीसरी बार", r"again and again", r"बार-बार")
GREETING_ONLY = _rx(r"^(?:hi+|hello+|hey|namaste|namaskar|नमस्ते|नमस्कार|good (?:morning|afternoon|evening))[\s!,.]*$")

def detect_topic(text: str) -> str:
    for name, rx in TOPICS:
        if rx.search(text): return name
    return "other"

def heuristic_flags(text: str, session_has_order: bool = False) -> dict:
    t = _nfd(text.strip())
    topic = detect_topic(t)
    only_numbers = len("".join(digit_runs(t))) >= max(4, int(0.5 * len(re.sub(r"\W", "", t)) or 1)) and topic == "other"
    has_inline_data = bool(parse_dates(t)) or bool(detect_payment_method(t) and re.search(r"\d", t)) or bool(INLINE_STATUS.search(t))
    personal = bool(ORDER_SPECIFIC.search(t))
    asks_status = topic in ("refund", "delivery", "cancel", "damage", "return")
    store_pickup_question = bool(re.search(r"(?:pick\s*up|pickup|collect).{0,35}(?:my\s+order|order).{0,35}(?:store|shop|near me)|(?:store|shop).{0,25}(?:pickup|pick up|collect)", t, re.I))
    generic_policy = bool(re.search(
        r"(?:cash on delivery|cod|wallet|bank account|expire|cancel.*(?:one item|item)|delivery address|standard delivery|express delivery|plus membership|refund.*after.*initiat|coupon.*(?:fail|apply)|exchange.*(?:size|colour|color)|not placed.*(?:money|order)|order.*not placed|money was deducted|warranty.*cover|no ticket|every metro|metro city|delivered.*never.*received|never.*received.*order|order says delivered)",
        t, re.I
    ))
    needs_order = (personal and not has_inline_data) or (bool(MENTIONS_ORDER.search(t)) and topic == "other" and not only_numbers
                                                         and not GREETING_ONLY.search(t) and bool(re.search(r"help|madad|मदद|baare|बारे|status|check", t, re.I)))
    if generic_policy and not re.search(r"(?:where is my|status of my|track my|refund status|where.*my order|my order.*(?:status|tracking|where))", t, re.I):
        needs_order = False
    if store_pickup_question: needs_order = False
    if session_has_order and asks_status and not has_inline_data: needs_order = True
    substantive = (topic != "other" or "?" in t or "कैसे" in t or "क्या" in t) and not only_numbers
    query = " ".join(x for x in [t, {"damage": "damaged wrong missing item report", "refund": "refund timeline", "return": "return window eligibility",
                                      "delivery": "delivery late tracking", "cancel": "cancel order", "warranty": "warranty",
                                      "payment": "payment", "coupon": "coupon", "plus": "plus membership"}.get(topic, "")] if x)
    return {"topic": topic, "needs_order": needs_order, "substantive": substantive,
            "human_request": bool(HUMAN.search(t)), "legal": bool(LEGAL.search(t)), "safety": bool(SAFETY.search(t)),
            "unrecognized_activity": bool(UNRECOGNIZED.search(t)), "compensation_demand": bool(COMPENSATION.search(t)),
            "cancel_request": bool(CANCEL.search(t)), "third_party_request": bool(THIRD_PARTY.search(t)), "repeat_contact": bool(THIRD_TIME.search(t)),
            "search_query": query, "only_numbers": only_numbers, "has_inline_data": has_inline_data}
