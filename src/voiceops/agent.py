"""ZentroMart support agent. Code owns: identity check, escalation rules, dates/amounts, tool calls.
The LLM owns: understanding the customer and wording the answer from retrieved policy + computed facts."""
import re, time
from dataclasses import dataclass, field
from . import calc
from .facts import order_facts, inline_facts, load_products, find_products, base_title, date_variants, fmt, HI_MONTHS, REFUND_TIMING
from .intent import heuristic_flags, INLINE_STATUS
from .kb import Retriever, tokenize
from .llm import get_llm, parse_json, StubLLM
from .prompts import SYSTEM, ANALYSIS, T, is_localized
from .textnorm import detect_language, extract_ids, spoken_to_digits, parse_dates, detect_payment_method
from .tools_adapter import Toolbox

REASONS = {
    "human_request": "ESC-1.6 customer asked for a human agent",
    "legal": "ESC-1.3 customer mentioned legal action, consumer court or a police complaint",
    "safety": "ESC-1.7 product safety risk (swelling battery, burning smell, sparks or injury)",
    "unrecognized_activity": "ESC-1.2 customer reports an order, payment or account activity they did not make",
    "repeat_contact": "ESC-1.4 third or later contact about this order within 7 days (contacts_last_7_days >= 2)",
    "refund_overdue": "ESC-1.5 refund is more than 3 business days past its maximum timeline",
    "high_value_damage": "ESC-1.1 damaged, defective or wrong item worth more than Rs 10,000",
}
SHIPPED_NOT_DELIVERED = {"shipped", "out_for_delivery", "delivery_attempted"}
LANG_NAMES = {"en": "en (English)", "hi": "hi (Hindi, Devanagari script)", "hi-en": "hi-en (Hinglish: Hindi written in Latin script)",
              "ta": "ta (Tamil)", "te": "te (Telugu)", "kn": "kn (Kannada)", "ml": "ml (Malayalam)", "bn": "bn (Bengali)",
              "gu": "gu (Gujarati)", "mr": "mr (Marathi)", "pa": "pa (Punjabi)"}
TOPICS_OK = {"refund", "return", "cancel", "delivery", "damage", "warranty", "payment", "coupon", "plus", "troubleshoot", "other"}
_BOOL = lambda v: v if isinstance(v, bool) else str(v).strip().lower() in ("true", "yes", "1")

@dataclass
class Session:
    lang: str = "en"
    turns: int = 0
    intro_done: bool = False
    order_id: str = None
    verified: bool = False
    order: dict = None
    attempts: int = 0
    locked: bool = False
    pending_question: str = None
    pending_flags: dict = None
    pending_verifier: str = None
    history: list = field(default_factory=list)
    tickets: list = field(default_factory=list)

class Agent:
    def __init__(self, root=None, llm=None, channel="chat", use_llm=True):
        from .config import dataset_dir
        self.root = root or dataset_dir()
        self.retriever = Retriever(self.root)
        self.products = load_products(self.root)
        self.holidays = calc.load_holidays(self.root)
        self.tools = Toolbox(self.root)
        self.llm = (llm if llm is not None else get_llm()) if use_llm else StubLLM()
        self.channel = channel
        self.llm_errors = 0; self.last_llm_error = ""
    @property
    def llm_on(self): return bool(getattr(self.llm, "available", False))
    def new_session(self): return Session()

    # ------------------------------------------------------------------ public
    def respond(self, sess: Session, text: str) -> dict:
        t0 = time.time(); self.tools.calls = []
        text = (text or "").strip(); sess.turns += 1
        if not text:
            return self._finish(sess, text, T("unclear", sess.lang), "clarify", sess.lang, {}, [], t0, templated=True)
        lang = detect_language(text, sess.lang if sess.turns > 1 else "en")
        flags = self._analyze(sess, text)
        if flags.get("language") and sum(c.isalpha() for c in text) >= 4: lang = flags["language"]
        sess.lang = lang
        ids = extract_ids(text, expect_verifier=bool(sess.order_id and not sess.verified))

        hard = next((k for k in ("human_request", "legal", "safety", "unrecognized_activity") if flags.get(k)), None)
        if hard: return self._handover(sess, text, lang, hard, flags, t0)
        if flags.get("third_party_request"):
            chunks = self._retrieve(sess, text, flags, sess.order if sess.verified else None, [])
            return self._finish(sess, text, T("third_party", lang), "refuse", lang, flags, chunks, t0, templated=True)

        if ids["order_id"] and ids["order_id"] != sess.order_id:
            sess.order_id, sess.verified, sess.order, sess.attempts, sess.locked = ids["order_id"], False, None, 0, False
            self.tools.lookup_order(sess.order_id)          # no proof yet: returns verification_required only
        if ids["verifier"] and not sess.order_id: sess.pending_verifier = ids["verifier"]
        ver = ids["verifier"] or (sess.pending_verifier if ids["order_id"] else None)

        reply = decision = None; just_verified = False
        if sess.order_id and not sess.verified and not sess.locked and ver:
            res = self.tools.lookup_order(sess.order_id, ver)
            if res.get("verified"):
                sess.verified, sess.order, just_verified, sess.pending_verifier = True, res, True, None
            else:
                sess.attempts += 1
                if sess.attempts >= 2: sess.locked = True; reply, decision = T("locked", lang), "refuse"
                else: reply, decision = T("retry", lang), "clarify"

        in_order_flow = flags["needs_order"] or bool(ids["order_id"]) or bool(ids["verifier"])
        if reply is None and sess.locked and flags["needs_order"]:
            reply, decision = T("locked", lang), "refuse"
        if reply is None and not sess.verified and not sess.locked and in_order_flow:
            if flags.get("pending_ok", flags["substantive"]) and flags["needs_order"]: sess.pending_question, sess.pending_flags = text, flags
            reply, decision = (T("ask_verifier", lang), "clarify") if sess.order_id else (T("ask_order", lang), "clarify")

        eff_text, eff_flags = text, flags
        if reply is None and just_verified:
            if sess.pending_question:
                eff_text, eff_flags, sess.pending_question = sess.pending_question, sess.pending_flags, None
            elif not flags["substantive"]:
                reply, decision = T("verified", lang), "answer"
        if reply is not None:
            evidence_flags = dict(flags)
            if ids["verifier"] or just_verified:
                evidence_flags["verification_step"] = True
            chunks = self._retrieve(sess, text, evidence_flags, sess.order if sess.verified else None, [])
            return self._finish(sess, text, reply, decision, lang, flags, chunks, t0, templated=True)
        if just_verified:
            eff_flags = {**eff_flags, "verification_step": True}
        return self._answer(sess, eff_text, eff_flags, lang, t0)

    def respond_prompt_only(self, text: str) -> str:
        """Step 1 baseline: role prompt only. No documents, no tools."""
        system = ("You are ZentroMart's AI support assistant for an Indian online store that sells groceries and electronics. "
                  "In your first reply tell the customer you are an AI assistant. Help with orders, deliveries, returns, refunds and cancellations. "
                  "Be polite, never share personal data, keep replies to 2 or 3 short sentences and reply in the customer's language.")
        return self.llm.complete(system, [{"role": "user", "content": text}], max_tokens=300)

    # ------------------------------------------------------------------ analysis
    def _analyze(self, sess, text):
        """Keyword rules are reliable when a word is explicit; the LLM only resolves what they cannot (follow-ups, other languages)."""
        h = heuristic_flags(text, session_has_order=sess.verified)
        if not self.llm_on or h["only_numbers"] or (h["topic"] == "other" and len(text) < 45 and re.search(r"\d", text)): return h
        hist = "\n".join(f"{r}: {m}" for r, m in sess.history[-6:]) or "(none)"
        prompt = f"CONVERSATION SO FAR:\n{hist}\n\nLATEST CUSTOMER MESSAGE:\n{text}\n\nIdentity verified in this session: {sess.verified}"
        try: js = parse_json(self.llm.complete(ANALYSIS, [{"role": "user", "content": prompt}], max_tokens=500))
        except Exception as e: self.llm_errors += 1; self.last_llm_error = str(e)[:300]; return h
        if not isinstance(js, dict): return h
        f, ids = dict(h), extract_ids(text, expect_verifier=bool(sess.order_id and not sess.verified))
        explicit = h["topic"] != "other" or h["has_inline_data"]
        llm_topic = str(js.get("topic", "other")).lower()
        f["topic"] = h["topic"] if h["topic"] != "other" else (llm_topic if llm_topic in TOPICS_OK else "other")
        f["needs_order"] = h["needs_order"] if explicit else _BOOL(js.get("needs_order", h["needs_order"]))
        f["substantive"] = h["substantive"] if (ids["order_id"] or ids["verifier"]) else (h["substantive"] or _BOOL(js.get("substantive", False)))
        f["pending_ok"] = h["substantive"] or (_BOOL(js.get("substantive", False)) and f["topic"] != "other")   # a generic "I need help" is not a question to remember
        f["cancel_request"] = h["cancel_request"] or _BOOL(js.get("cancel_request", False))
        f["third_party_request"] = h["third_party_request"] or _BOOL(js.get("third_party_request", False))
        for k in ("human_request", "legal", "safety", "unrecognized_activity"): f[k] = _BOOL(js.get(k, False)) or h[k]
        if js.get("search_query"): f["search_query"] = str(js["search_query"])
        if js.get("language") in LANG_NAMES: f["language"] = js["language"]
        return f

    # ------------------------------------------------------------------ answering
    def _answer(self, sess, text, flags, lang, t0):
        order = sess.order if sess.verified else None
        facts, info = [], None
        if order:
            facts, info = order_facts(order, self.holidays, self.products)
            if not flags.get("only_numbers"):
                if info["contacts"] >= 2: return self._handover(sess, text, lang, "repeat_contact", flags, t0)
                if flags["topic"] == "refund" and info["overdue"]:
                    return self._handover(sess, text, lang, "refund_overdue", flags, t0, rrn=info["overdue"][0]["rrn"])
                if flags["topic"] == "damage" and info["damage_present"] and info["damage_high_value"]:
                    return self._handover(sess, text, lang, "high_value_damage", flags, t0)
        norm_text = spoken_to_digits(text)
        inl_lines, inl = inline_facts(text, flags["topic"], self.holidays, self.products)
        facts = facts + inl_lines
        if inl["high_value_damage"] and not order: return self._handover(sess, text, lang, "high_value_damage", flags, t0)
        if inl["overdue"] and not order: return self._handover(sess, text, lang, "refund_overdue", flags, t0)

        chunks = self._retrieve(sess, norm_text, flags, order, inl["products"])
        decision, hint, forced = "answer", "", False
        if flags.get("compensation_demand"):
            hint = ("The customer is demanding compensation. If a policy in the context gives them something specific (for example the Express late-delivery credit) "
                    "state exactly that and use decision answer. Otherwise use decision refuse: apologise, say policy does not allow compensation or vouchers here, promise nothing, offer other help.")
        elif flags.get("cancel_request") and order and order["status"] in SHIPPED_NOT_DELIVERED:
            decision, forced = "refuse", True
            if (order.get("payment_method") or "").lower() == "cod":
                refund_hint = " For cash-on-delivery, state that nothing is charged if the customer refuses the parcel."
            else:
                refund_hint = (" For this prepaid order, explain that if delivery is refused, the refund is initiated once the parcel reaches "
                               "the warehouse, usually 7 to 10 calendar days after the last delivery attempt (REF-1). Do not say that this is "
                               "the time the bank takes to credit the refund.")
            hint = ("refuse. The order has already shipped, so it cannot be cancelled. Say the customer can refuse the delivery at the door, "
                    "or return the item after delivery, and state the return window in days from FACTS." + refund_hint)
        elif flags.get("cancel_request") and (bool(order and order["status"] in SHIPPED_NOT_DELIVERED) or bool(INLINE_STATUS.search(norm_text))):
            # A general question about shipped-order policy is an answer; a verified order-specific cancellation is a refusal.
            decision, forced = ("refuse" if order and order["status"] in SHIPPED_NOT_DELIVERED else "answer"), True
            prepaid_hint = ""
            if order and order["status"] in SHIPPED_NOT_DELIVERED:
                if (order.get("payment_method") or "").lower() == "cod":
                    prepaid_hint = " For this cash-on-delivery order, say nothing is charged if the parcel is refused."
                else:
                    prepaid_hint = (" For this prepaid order, if the customer refuses delivery, the refund is initiated once the parcel reaches "
                                    "the warehouse, usually 7 to 10 calendar days after the last delivery attempt (REF-1); this is not the bank-credit timeline.")
            hint = ("The order has already shipped or is on its way, so it cannot be cancelled. "
                    "Say they may refuse the parcel at the door or accept delivery and return eligible items within the applicable return window. "
                    "Do not claim to cancel the order. Use decision refuse only when this is a verified order-specific cancellation request; otherwise answer the policy question." + prepaid_hint)
        elif flags.get("cancel_request") and order:
            hint = "answer. You cannot cancel it yourself: tell the customer how to cancel in My Orders and what the refund timing is for their payment method."
        reply = None
        if self.llm_on:
            try: reply, d2 = self._generate(sess, text, flags, lang, chunks, facts, hint)
            except Exception as e: self.llm_errors += 1; self.last_llm_error = str(e)[:300]
            else:
                if not forced and d2 in ("answer", "clarify", "refuse"): decision = d2
        if reply is None:
            if flags.get("cancel_request") and (bool(order and order["status"] in SHIPPED_NOT_DELIVERED) or bool(INLINE_STATUS.search(norm_text))):
                category_windows = []
                category_names = {"electronics": "electronics", "fashion": "clothing/footwear/bags", "home_kitchen": "home/kitchen", "beauty_sealed": "sealed beauty/personal-care", "books_media": "books/stationery"}
                if order:
                    for item in order.get("items", []):
                        days = (info or {}).get("return_windows", {}).get(item.get("sku"))
                        cat = item.get("return_category")
                        if days and cat:
                            phrase = f"{days} days for {category_names.get(cat, cat)}"
                            if phrase not in category_windows: category_windows.append(phrase)
                window_text = (" The return window depends on the item category: " + "; ".join(category_windows) + ".") if category_windows else ""
                payment_method = (order or {}).get("payment_method", "").lower()
                refund_text = ""
                if order and order.get("status") in SHIPPED_NOT_DELIVERED:
                    if payment_method == "cod":
                        refund_text = {
                            "en": "This is cash on delivery, so nothing is charged if you refuse the parcel.",
                            "hi": "यह कैश-ऑन-डिलीवरी ऑर्डर है, इसलिए पार्सल लेने से मना करने पर कोई शुल्क नहीं लिया जाएगा।",
                            "hi-en": "Yeh cash-on-delivery order hai, isliye parcel refuse karne par koi charge nahi lagega.",
                        }.get(lang, "This is cash on delivery, so nothing is charged if you refuse the parcel.")
                    else:
                        refund_text = {
                            "en": "If you refuse this prepaid parcel, the refund is initiated once it reaches the warehouse, usually 7 to 10 calendar days after the last delivery attempt.",
                            "hi": "यदि आप प्रीपेड पार्सल लेने से मना करते हैं, तो पार्सल वेयरहाउस पहुँचने पर रिफंड शुरू होता है, आमतौर पर अंतिम डिलीवरी प्रयास के 7 से 10 कैलेंडर दिन बाद।",
                            "hi-en": "Agar aap yeh prepaid parcel refuse karte hain, toh parcel warehouse pahunchne par refund initiate hota hai, aam taur par last delivery attempt ke 7 se 10 calendar days baad.",
                        }.get(lang, "If you refuse this prepaid parcel, the refund is initiated once it reaches the warehouse, usually 7 to 10 calendar days after the last delivery attempt.")
                reply = T("shipped_cancel", lang, window_text=window_text, refund_text=refund_text)
                decision = "refuse" if order and order["status"] in SHIPPED_NOT_DELIVERED else "answer"
            else:
                # Vague duration questions lack the product category needed for a precise return window.
                if (flags["topic"] == "return" and not order and not inl.get("products")
                        and re.search(r"how\s+many\s+days|how\s+long|kitne\s+din|कितने\s+दिन", text, re.I)
                        and not re.search(r"electronics|fashion|clothing|footwear|home|kitchen|beauty|books|stationery|grocery|innerwear|category", text, re.I)):
                    reply, decision = T("return_clarify", lang), "clarify"
                else:
                    reply = self._extractive(chunks, facts, flags, lang, text=text, order=order, info=info)
            if flags.get("compensation_demand"): reply, decision = T("refuse_comp", lang), "refuse"
        if decision == "refuse" and not re.search(r"sorry|apolog|maaf|क्षमा|खेद|माफ", reply, re.I): reply = T("apology", lang) + " " + reply
        return self._finish(sess, text, reply, decision, lang, flags, chunks, t0, musts=self._musts(flags, info, inl, order, text))

    def _forced_ids(self, flags, order, text, prods):
        topic, f = flags["topic"], []
        cats = {p["return_category"] for p in prods} | ({i["return_category"] for i in order["items"]} if order else set())
        status = order["status"] if order else None
        if topic == "refund":
            f.append("REF-2")
            if order and order.get("payment_method") == "cod": f.append("REF-3")
            if order and order.get("return_status"):
                f.append("RET-7")
                if not order.get("is_plus"): f.append("REF-1")
            if order and order.get("is_plus"): f.append("PLS-2")
        elif topic == "return":
            f += ["RET-2"] + (["RET-3"] if "electronics" in cats else []) + (["RET-8"] if "fashion" in cats else [])
            if "fashion" in cats: f.append("RET-6")
            if re.search(r"open|used|tried|worn|seal|खोल|इस्तेमाल|istemal|kholkar", text, re.I): f.append("RET-4")
        elif topic == "cancel":
            f += ["CAN-2"] if status in SHIPPED_NOT_DELIVERED else (["CAN-1"] if order else ["CAN-1", "CAN-2"])
            if order and order.get("delivery_type") == "slot": f.append("CAN-3")
        elif topic == "delivery":
            if re.search(r"deliver\s+to|delivery\s+to|ship\s+to|outside\s+india|international|dubai", text, re.I):
                f += ["DEL-5"]
            else:
                f += ["TRK-1"] + (["DEL-4"] if re.search(r"late|delay|देर|der", text, re.I) else ["DEL-1"])
            if re.search(r"not at home|not home|not there|ghar (?:pe|par) nahi|घर पर नहीं|miss", text, re.I): f.append("DEL-3")
        elif topic == "damage":
            f += ["DWM-1"]
            if order and order.get("damage_hours_after_delivery") and int(order["damage_hours_after_delivery"]) > 48: f.append("DWM-4")
            if any(float(p.get("price_inr") or 0) > 10000 for p in prods):
                f += ["DWM-3", "ESC-1"]
        elif topic == "warranty":
            f += ["WAR-1", "WAR-2"] + (["MKS-3"] if any(p["fulfilled_by"] != "ZentroMart" for p in prods) else [])
        if flags.get("third_party_request"): f += ["ESC-4", "IDV-4"]
        if flags.get("verification_step"): f.insert(0, "IDV-2")
        reason = flags.get("handover_reason")
        if reason == "high_value_damage": f += ["DWM-3", "ESC-1"]
        elif reason: f.append("ESC-1")
        return f[:3]

    def _retrieve(self, sess, text, flags, order, prods):
        sq = flags.get("search_query") or ""
        tt, st = set(tokenize(text)), set(tokenize(sq))
        q = text if not sq else (sq if (tt and len(tt & st) / len(tt) >= 0.7) else f"{text} {sq}")   # keep the customer's own words (SKUs, product names, Hindi) without doubling them
        if len(text.split()) < 7:
            prev = [m for r, m in sess.history if r == "customer"][-1:]
            if prev: q += " " + prev[0]
        res = self.retriever.search(q, k=4, catalog_k=0)
        have = {r["id"] for r in res}
        extra = [self.retriever.by_id[i] for i in self._forced_ids(flags, order, text, prods) if i in self.retriever.by_id and i not in have]
        extra += [self.retriever.by_id[f"products.csv:{p['sku']}"] for p in prods if f"products.csv:{p['sku']}" in self.retriever.by_id]
        return res + [dict(score=0.0, attached=True, **c) for c in extra]

    def _generate(self, sess, text, flags, lang, chunks, facts, hint):
        flat = lambda t: re.sub(r"\s+", " ", t)[:800]
        ctx = "\n".join(f"[{c['id']}] ({c['doc_type']}, {c['status']}{', SUPERSEDED - do not use' if c['status'] == 'superseded' else ''}) {c['title']}: "
                        f"{flat(c['text'])}" for c in chunks) or "(no matching policy sections found)"
        hist = "\n".join(f"{r}: {m}" for r, m in sess.history[-6:]) or "(none)"
        tools = "; ".join(f"{c['tool']}({', '.join(f'{k}={v}' for k, v in c['args'].items() if k != 'verification_value')}) ok={c['ok']}" for c in self.tools.calls) or "(none this turn)"
        prompt = (f"LANGUAGE: {LANG_NAMES.get(lang, lang)}\nCHANNEL: {'voice call (short spoken sentences)' if self.channel == 'voice' else 'chat'}\n"
                  f"CUSTOMER IDENTITY VERIFIED: {sess.verified}\n\nCONVERSATION SO FAR:\n{hist}\n\n"
                  f"FACTS (computed by code, trust them):\n{chr(10).join(facts) if facts else '(no order or inline facts)'}\n\n"
                  f"TOOL RESULTS: {tools}\n\nPOLICY CONTEXT:\n{ctx}\n\nDECISION HINT: {hint or 'none'}\n\n"
                  f"LATEST CUSTOMER MESSAGE: {text}\n\nRespond with the JSON object only.")
        out = self.llm.complete(SYSTEM, [{"role": "user", "content": prompt}], max_tokens=1000)
        js = parse_json(out)
        if isinstance(js, dict) and js.get("reply"): return str(js["reply"]).strip(), str(js.get("decision", "answer")).lower()
        m = re.search(r'"reply"\s*:\s*"((?:[^"\\]|\\.)*)', out or "", re.S)       # truncated / malformed JSON: salvage the reply text
        if m: return m.group(1).replace('\\"', '"').replace("\\n", " ").strip(), "answer"
        return (out or "").strip(), "answer"

    def _extractive(self, chunks, facts, flags, lang, text="", order=None, info=None):
        """Offline fallback: use deterministic facts and the policy section relevant to the intent."""
        def clean(s): return re.sub(r"\s+", " ", re.sub(r"[*|#>]|^\s*[-\d.]+\s", " ", s)).strip()
        topic, parts = flags["topic"], []
        if topic == "other":
            return T("no_info", lang)

        calc_lines = [f.replace("CALCULATED: ", "") for f in facts if f.startswith("CALCULATED")]
        policy_lines = [f.split("POLICY ANSWER: ", 1)[1] for f in facts if f.startswith("POLICY ANSWER: ")]
        if calc_lines: parts.append(calc_lines[0][0].upper() + calc_lines[0][1:] + ".")
        if policy_lines:
            parts.extend(policy_lines[:2])

        if topic == "delivery" and order:
            date_line = next((f for f in facts if f.startswith("Dates:")), None)
            if date_line: parts.insert(0, clean(date_line))
        if topic == "refund" and order:
            order_refund_facts = [f for f in facts if f.startswith(("Refund timing for this order's payment method", "Return refund:", "Cancellation refund:", "Returned-to-origin refund:", "Refund process after a return pickup:", "Express late-delivery credit:"))]
            for f in order_refund_facts:
                if not any(clean(f).lower() in p.lower() for p in parts): parts.append(clean(f))

        # A return pickup refund is not initiated until QC passes, unless Plus rules apply.
        if topic == "refund" and order and order.get("return_status") and not order.get("return_refund_initiated_on"):
            qc = next((f for f in facts if f.startswith("Refund process after a return pickup:")), None)
            if qc and not any("quality check" in p.lower() or "qc" in p.lower() for p in parts): parts.append(clean(qc))

        pick = {
            "return": ("RET-4", "RET-5", "RET-2", "RET-6", "RET-8"),
            "refund": ("RET-7", "REF-3", "REF-2", "REF-1", "REF-6", "REF-4", "REF-5"),
            "delivery": ("DEL-3", "DEL-4", "DEL-1", "DEL-2", "DEL-5", "DEL-6"),
            "damage": ("DWM-1", "DWM-2", "DWM-3", "DWM-4", "DWM-5"),
            "warranty": ("WAR-2", "WAR-1", "WAR-3"),
            "cancel": ("CAN-2", "CAN-3", "CAN-1", "FAQH-1"),
            "troubleshoot": (),
            "payment": ("PAY-1", "PAY-2", "REF-2", "REF-3"),
            "coupon": ("CPN-1", "CPN-2"),
            "plus": ("PLS-1", "PLS-2"),
        }.get(topic, ())
        by_id = {c["id"]: c for c in chunks}
        if topic == "return" and re.search(r"fee|charge|collection|pickup|pick-up|शुल्क|फीस", text, re.I):
            pick = ("RET-6", "RET-2", "RET-4", "RET-5")
        if topic == "delivery" and re.search(r"deliver\s+to|delivery\s+to|ship\s+to|outside\s+india|international|dubai", text, re.I):
            pick = ("DEL-5", "DEL-1", "DEL-2")
        elif topic == "delivery" and re.search(r"midnight|schedule.{0,20}deliver|deliver.{0,20}midnight", text, re.I):
            return T("no_info", lang)
        relevant = [by_id[i] for i in pick if i in by_id]
        if topic == "troubleshoot":
            relevant = [c for c in chunks if c["id"].startswith(("TRB-", "HTR-"))]
        if not relevant:
            relevant = [c for c in chunks if c["doc_type"] != "catalog" and not c["id"].startswith(("KB-00", "STY", "IDV")) and c["status"] != "superseded"]
        if relevant and len(parts) < 3:
            chosen = relevant[0]
            if topic == "troubleshoot":
                numbered = [re.sub(r"^\s*\d+\.\s*", "", line.strip()) for line in chosen["text"].splitlines() if re.match(r"^\s*\d+\.\s+", line)]
                candidate = " ".join(numbered[:3])
            else:
                sents = re.split(r"(?<=[.!?])\s+", clean(chosen["text"]))
                sentence_count = 3 if chosen["id"] == "RET-6" else 2
                candidate = " ".join(sents[:sentence_count])
            # Don't append a generic section when it contradicts computed facts or repeats product policy.
            if candidate and candidate.lower() not in " ".join(parts).lower(): parts.append(candidate)
        return " ".join(parts)[:620] if parts else T("no_info", lang)

    def _musts(self, flags, info, inl, order, text):
        """Computed facts that must appear in the reply: [(alternatives per required item, {lang: sentence})]. Code appends them if the LLM omitted them."""
        out, topic = [], flags["topic"]
        def hi(d): return f"{d.day} {HI_MONTHS[d.month - 1]} {d.year}"
        if (topic == "cancel" and order and order.get("status") in SHIPPED_NOT_DELIVERED
                and (order.get("payment_method") or "").lower() != "cod"):
            out.append(([["7 to 10", "7-10", "7–10", "7 se 10"]], {
                "en": "If you refuse this prepaid parcel, the refund is initiated once it reaches the warehouse, usually 7 to 10 calendar days after the last delivery attempt.",
                "hi-en": "Agar aap yeh prepaid parcel refuse karte hain, toh parcel warehouse pahunchne par refund initiate hota hai, aam taur par last delivery attempt ke 7 se 10 calendar days baad.",
                "hi": "यदि आप प्रीपेड पार्सल लेने से मना करते हैं, तो पार्सल वेयरहाउस पहुँचने पर रिफंड शुरू होता है, आमतौर पर अंतिम डिलीवरी प्रयास के 7 से 10 कैलेंडर दिन बाद।"
            }))
        if topic == "refund":
            d = inl.get("refund_latest")
            if not d and info:
                pend = [r for r in info["refunds"] if not r["completed"] and not r["overdue"] and r["min_date"] != r["max_date"]]
                d = pend[0]["max_date"] if len(pend) == 1 else None
            if d: out.append(([date_variants(d)], {"en": f"The latest expected date is {fmt(d)}.", "hi-en": f"Latest expected date {fmt(d)} hai.", "hi": f"अपेक्षित अंतिम तिथि {hi(d)} है।"}))
        if topic == "return":
            dl, days = inl.get("deadline"), inl.get("window_days")
            if not dl and order and info and info["return_deadlines"]:
                low = text.lower(); named = [i["sku"] for i in order["items"] if i["sku"] in text.upper() or base_title(i["title"]) in low]
                sku = named[0] if named else (list(info["return_deadlines"])[0] if len(info["return_deadlines"]) == 1 else None)
                if sku in info["return_deadlines"]: dl, days = info["return_deadlines"][sku], info["return_windows"].get(sku)
            if dl and days:
                out.append(([[str(days)], date_variants(dl)], {"en": f"The return window is {days} days, so the last date is {fmt(dl)}.",
                           "hi-en": f"Return window {days} din ka hai, last date {fmt(dl)} hai.", "hi": f"वापसी की अवधि {days} दिन है, अंतिम तारीख {hi(dl)} है।"}))
            prods = inl.get("products", [])
            if prods:
                p = prods[0]; cat = p["return_category"]
                if cat in calc.NOT_RETURNABLE:
                    out.append(([ ["cannot", "can't", "not returnable", "non-returnable", "नहीं", "nahi", "nahin", "not be returned"] ],
                                {"en": "This category is not returnable for change of mind.", "hi-en": "Yeh category change-of-mind return ke liye returnable nahi hai.", "hi": "यह श्रेणी मन बदलने पर वापसी के लिए योग्य नहीं है।"}))
                else:
                    normal_days = inl.get("window_days") or calc.RETURN_WINDOWS.get(cat)
                    if cat == "electronics": normal_days = normal_days or calc.RETURN_WINDOWS["electronics"]
                    if cat == "fashion":
                        qdates = parse_dates(text)
                        festive_order = any(calc.FESTIVE_START <= d <= calc.FESTIVE_END for d in qdates) and bool(re.search(r"sale|bought|buy|ordered|order placed|purchased|kiya|order kiya", text, re.I))
                        normal_days = 30 if festive_order else 15
                    if normal_days:
                        out.append(([[str(normal_days)]], {"en": f"The return window is {normal_days} calendar days from delivery.",
                                    "hi-en": f"Return window {normal_days} calendar days ka hai, delivery date se.", "hi": f"वापसी की अवधि डिलीवरी की तारीख से {normal_days} कैलेंडर दिन है।"}))
                    if cat == "beauty_sealed":
                        out.append(([["unopened", "seal", "sealed"]], {"en": "Beauty products are returnable only if unopened with the seal intact.",
                                    "hi-en": "Beauty product tabhi return ho sakta hai jab unopened ho aur seal intact ho.", "hi": "ब्यूटी उत्पाद तभी लौटाया जा सकता है जब वह बंद हो और सील सही हो।"}))
                        if re.search(r"open|opened|used|tried|khol|खोल|इस्तेमाल", text, re.I):
                            out.append(([["cannot", "can't", "not returnable", "non-returnable", "नहीं", "nahi", "nahin"]],
                                        {"en": "Because it has been opened or tried, it does not meet the sealed condition for a change-of-mind return.",
                                         "hi-en": "Product khola ya try kiya gaya hai, isliye change-of-mind return ki sealed condition poori nahi hoti.",
                                         "hi": "इसे खोलने या इस्तेमाल करने के कारण यह सामान्य वापसी की सील वाली शर्त पूरी नहीं करता।"}))
        if topic == "warranty" and inl.get("products"):
            p = inl["products"][0]
            assured = str(p.get("assured", "")).lower() == "true" and p.get("fulfilled_by") == "ZentroMart"
            if assured:
                out.append(([ ["12"], ["My Orders", "app"] ], {"en": "This product has a 12-month ZentroMart Assured warranty from delivery in addition to its brand warranty. Raise claims in My Orders in the app; ZentroMart arranges pickup and repair or replacement.",
                            "hi-en": "Is product par delivery se 12-month ZentroMart Assured warranty brand warranty ke alawa milti hai. Claim app ke My Orders mein raise karein.",
                            "hi": "इस उत्पाद पर ब्रांड वारंटी के अतिरिक्त डिलीवरी से 12 महीने की ZentroMart Assured वारंटी है। दावा ऐप के My Orders में करें।"}))
            else:
                months = int(p.get("warranty_months") or 0)
                if months:
                    out.append(([[str(months)]], {"en": f"The catalog lists a {months}-month brand warranty; brand warranty claims are made with the manufacturer's service centre.",
                                "hi-en": f"Catalog mein {months}-month brand warranty listed hai; claim manufacturer ke service centre mein karein.",
                                "hi": f"कैटलॉग में {months} महीने की ब्रांड वारंटी है; दावा निर्माता के सर्विस सेंटर में करें।"}))
        if topic == "refund":
            method = detect_payment_method(text)
            if method and not inl.get("refund_latest"):
                if method == "upi": must = [["1"], ["3"]]
                elif method in ("card", "cod"): must = [["5"], ["7"]] if method == "card" else [["ZentroMart Wallet", "wallet"], ["5"], ["7"]]
                elif method == "netbanking": must = [["3"], ["5"]]
                else: must = [["instant", "2 hours", "two hours", "तुरंत", "turant"]]
                out.append((must, {"en": f"{REFUND_TIMING[method]}.", "hi-en": f"{REFUND_TIMING[method]}.", "hi": f"{REFUND_TIMING[method]}."}))
        if topic == "damage" and inl.get("products") and not inl.get("high_value_damage"):
            out.append(([["free", "no fee", "waived", "no charge"]], {"en": "A verified damaged-item claim gets a free replacement or full refund including delivery fees; no reverse pickup fee applies.",
                        "hi-en": "Verified damaged item par free replacement ya full refund mil sakta hai; reverse pickup fee nahi lagti.",
                        "hi": "सत्यापित क्षतिग्रस्त वस्तु पर मुफ्त प्रतिस्थापन या पूरा रिफंड मिल सकता है; रिवर्स पिकअप शुल्क नहीं है।"}))
        return out

    # ------------------------------------------------------------------ handover & output
    def _handover(self, sess, text, lang, reason, flags, t0, rrn=None):
        prods = find_products(text, self.products, allow_prefix_fuzzy=(flags.get("topic") == "return"))
        evidence_flags = {**flags, "handover_reason": reason}
        chunks = self._retrieve(sess, text, evidence_flags, sess.order if sess.verified else None, prods)
        earlier = " | ".join([m for r, m in sess.history if r == "customer"][-3:] + [text])[:320]
        checked = f"identity verified={sess.verified}"
        if sess.order: checked += f"; order status={sess.order.get('status')}; contacts_last_7_days={sess.order.get('contacts_last_7_days')}"
        summary = (f"Customer issue: {text[:240]}. Detected intent: {flags.get('topic', 'other')}. Order ID: {sess.order_id or 'not provided'}. "
                   f"What the customer told us: {earlier}. What we checked: {checked}. Reason for escalation: {REASONS[reason]}.")
        res = self.tools.create_ticket(summary, order_id=sess.order_id, reason=reason, intent=flags.get("topic", "other"))
        if res.get("created"):
            sess.tickets.append(res["ticket_id"])
            parts = ([T("safety_extra", lang)] if reason == "safety" else []) + [T("handover", lang)]
            if reason == "refund_overdue" and rrn: parts.append(T("overdue_extra", lang, rrn=rrn))
            reply = " ".join(parts)
        else:
            reply = T("handover_fail", lang)
        return self._finish(sess, text, reply, "escalate", lang, flags, chunks, t0, templated=True)

    def _localize(self, text, lang):
        try:
            out = self.llm.complete("Translate the customer-support message into the requested Indian language. Keep names, numbers and IDs unchanged. Output only the translation.",
                                    [{"role": "user", "content": f"Language: {LANG_NAMES.get(lang, lang)}\nMessage: {text}"}], max_tokens=300)
            return out.strip() or text
        except Exception as e:
            self.llm_errors += 1; self.last_llm_error = str(e)[:300]; return text

    def _scrub(self, reply, sess):
        r = reply
        if sess.order and sess.order.get("delivery_pincode"): r = r.replace(str(sess.order["delivery_pincode"]), "your registered pincode")
        r = re.sub(r"\+\d{1,3}[\s-]?\d[\d\s-]{7,}\d", "[number removed]", r)
        r = re.sub(r"(?<!\d)\d{10}(?!\d)", "[number removed]", r)
        return re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[email removed]", r)

    def _finish(self, sess, user_text, reply, decision, lang, flags, chunks, t0, templated=False, musts=None):
        if templated and not is_localized(lang) and self.llm_on: reply = self._localize(reply, lang)
        reply = self._scrub(reply, sess)
        low = reply.translate(str.maketrans("०१२३४५६७८९", "0123456789")).lower()
        for alt_lists, sentences in (musts or []):
            if any(not any(a.lower() in low for a in alts) for alts in alt_lists):
                reply += " " + sentences.get(lang, sentences["en"])
        if not sess.intro_done:
            if not re.search(r"\bAI\b|एआई", reply): reply = T("intro", lang if is_localized(lang) else "en") + " " + reply
            sess.intro_done = True
        sess.history += [("customer", user_text), ("assistant", reply)]
        keep = ("topic", "needs_order", "substantive", "human_request", "legal", "safety", "unrecognized_activity", "compensation_demand", "cancel_request")
        return {"reply": reply, "decision": decision, "language": lang, "tool_calls": list(self.tools.calls),
                "chunk_ids": [c["id"] for c in chunks], "chunks": [{"id": c["id"], "title": c["title"], "doc_type": c["doc_type"]} for c in chunks],
                "flags": {k: flags.get(k) for k in keep if k in flags}, "verified": sess.verified, "tickets": list(sess.tickets),
                "llm": getattr(self.llm, "name", "offline") if self.llm_on else "offline", "latency_s": round(time.time() - t0, 3)}
