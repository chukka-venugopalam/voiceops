"""Facts computed by code (dates, deadlines, overdue flags, values). The LLM is told to trust these and never recompute."""
import csv, os, re
from difflib import SequenceMatcher
from datetime import date
from . import calc
from .config import dataset_dir
from .textnorm import parse_dates, detect_payment_method, spoken_to_digits

REFUND_TIMING = {"upi": "UPI refunds take 1 to 3 business days", "card": "card refunds take 5 to 7 business days",
                 "netbanking": "net banking refunds take 3 to 5 business days", "wallet": "Wallet refunds are instant, within 2 hours",
                 "cod": "COD refunds go to ZentroMart Wallet (instant, within 2 hours) or bank transfer (5 to 7 business days)"}
HI_MONTHS = ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"]

def fmt(d: date) -> str: return f"{d.day} {d.strftime('%B')} {d.year}"

def date_variants(d: date):
    return [fmt(d), d.isoformat(), f"{d.day} {HI_MONTHS[d.month - 1]} {d.year}", f"{d.day:02d} {d.strftime('%B')} {d.year}"]

def load_products(root=None) -> dict:
    root = root or dataset_dir()
    with open(os.path.join(root, "data", "products.csv"), encoding="utf-8", newline="") as f:
        return {r["sku"]: r for r in csv.DictReader(f)}

def base_title(title: str) -> str:
    return title.split(",")[0].strip().lower()

def _product_tokens(value: str):
    """Split letters and digits consistently so ASR and catalog model tokens align."""
    return re.findall(r"[a-z]+|[0-9]+", value.lower())


_SPOKEN_ELE_SKU = re.compile(
    r"\bele(?:\s*-\s*|\s+)((?:(?:zero|one|two|to|three|four|five|six|seven|eight|nine|\d+)\s*){2,4})(?=\s|$|[,.)])",
    re.I,
)
_SKU_DIGITS = {"zero": "0", "one": "1", "two": "2", "to": "2", "three": "3", "four": "4",
               "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
_CAPACITY_TWO = re.compile(
    r"\bto(?=\s+(?:zero|one|two|three|four|five|six|seven|eight|nine|\d+)\b"
    r"(?:\s+(?:zero|one|two|three|four|five|six|seven|eight|nine|\d+)\b)*"
    r"\s+(?:gb|gigabytes?)\b)",
    re.I,
)


def _normalize_spoken_sku(text: str) -> str:
    def replace(match):
        digits = "".join(_SKU_DIGITS.get(token.lower(), token) for token in re.findall(r"[a-z]+|\d+", match.group(1), re.I))
        return f"ELE-{digits}"
    return _SPOKEN_ELE_SKU.sub(replace, text)

def _normalize_spoken_numbers(text: str) -> str:
    text = _CAPACITY_TWO.sub("two", text)
    return spoken_to_digits(_normalize_spoken_sku(text))


def _fuzzy_token_match(a: str, b: str) -> bool:
    if a == b:
        return True
    # Never fuzzy-match model numbers / pack counts; only tolerate ASR spelling slips in words.
    if not a.isalpha() or not b.isalpha() or min(len(a), len(b)) < 5:
        return False
    return SequenceMatcher(None, a, b).ratio() >= 0.86


def _contains_product_phrase(query_tokens, title_tokens):
    if not title_tokens or len(title_tokens) > len(query_tokens):
        return False
    width = len(title_tokens)
    for start in range(len(query_tokens) - width + 1):
        window = query_tokens[start:start + width]
        if all(_fuzzy_token_match(a, b) for a, b in zip(title_tokens, window)):
            return True
    return False


def find_products(text: str, products: dict, limit=2, allow_prefix_fuzzy=False):
    """Match a product SKU/name, tolerating punctuation loss and cautious ASR spelling errors.

    Prefix-only matching is restricted to return-window questions where variants share the
    same category rule; it is not used for warranty/Assured eligibility.
    """
    normalized_text = _normalize_spoken_numbers(text)
    low, up, hits = normalized_text.lower(), normalized_text.upper(), []
    query_tokens = _product_tokens(normalized_text)
    query_norm = " ".join(query_tokens)
    exact_sku_hits = []
    matches = []
    for sku, product in products.items():
        title = product["title"]
        b = base_title(title)
        base_tokens = _product_tokens(b)
        title_tokens = _product_tokens(title)
        base_norm = " ".join(base_tokens)
        title_norm = " ".join(title_tokens)
        sku_prefix, sku_number = sku.split("-", 1)
        sku_pattern = rf"(?<![A-Z0-9]){re.escape(sku_prefix)}(?:\s*-\s*|\s+){re.escape(sku_number)}(?![0-9])"
        exact_sku = re.search(sku_pattern, up) is not None
        if exact_sku:
            exact_sku_hits.append(product)
            continue
        exact_title = bool(title_norm and (
            f" {title_norm} " in f" {query_norm} "
            or _contains_product_phrase(query_tokens, title_tokens)
        ))
        base_match = bool(base_norm and (
            f" {base_norm} " in f" {query_norm} "
            or _contains_product_phrase(query_tokens, base_tokens)
        ))
        specificity = 2 if exact_title else 1 if base_match else 0

        if not specificity and allow_prefix_fuzzy:
            # For storage variants like “Kestrel K7 Smartphone 256 GB”, model identity is
            # enough to determine the category-based return window; omit the capacity suffix.
            prefix = base_tokens[:3]
            has_capacity_suffix = len(title_tokens) >= 5 and title_tokens[-1] == "gb" and title_tokens[-2].isdigit()
            if has_capacity_suffix and len(prefix) == 3:
                if _contains_product_phrase(query_tokens, prefix):
                    specificity = 1
        if specificity:
            matches.append((specificity, product))

    if exact_sku_hits:
        return exact_sku_hits[:limit]
    if matches:
        best_specificity = max(specificity for specificity, _ in matches)
        hits = [product for specificity, product in matches if specificity == best_specificity]
    return hits[:limit]

def _d(x): return calc.to_date(x) if x else None

def _refund_line(label, init, method, amount, rrn, completed, H):
    if not init or not method: return None, None
    init_d = _d(init); lo, hi = calc.refund_eta(init_d, method, H)
    done = bool(completed)
    overdue = (not done) and calc.refund_overdue(init_d, method, H)
    s = f"{label} refund: initiated {fmt(init_d)} via {method}" + (f", amount Rs {amount}" if amount else "")
    s += f"; expected between {fmt(lo)} and {fmt(hi)}" if lo != hi else "; wallet refunds are instant (within 2 hours of initiation)"
    s += f"; completed on {fmt(_d(completed))}" if done else "; completion not recorded"
    if rrn: s += f"; reference number (RRN) {rrn}"
    if overdue: s += f"; OVERDUE: more than 3 business days past the maximum date {fmt(hi)} -> hand over to a human (ESC-1.5)"
    info = dict(label=label, method=method, initiated=init_d, max_date=hi, min_date=lo, rrn=rrn, overdue=overdue, completed=done)
    return s, info

def order_facts(order: dict, H, products: dict, today=calc.TODAY):
    """Returns (lines, info). Never includes phone, email, address or pincode (IDV-4)."""
    L, info = [], {"contacts": int(order.get("contacts_last_7_days") or 0), "refunds": [], "overdue": [], "return_deadlines": {}, "return_windows": {},
                   "damage_present": False, "damage_high_value": False, "status": order.get("status")}
    placed = str(order.get("placed_at") or "")[:10]
    L.append(f"Order {order['order_id']}: status={order['status']}; placed={placed}; delivery_type={order.get('delivery_type')}; "
             f"payment={order.get('payment_method')}; Plus member={order.get('is_plus')}; first name={order.get('customer_first_name')}")
    dates = [f"{lbl} {fmt(_d(order[k]))}" for k, lbl in (("promised_date", "promised delivery"), ("shipped_on", "shipped on"), ("delivered_on", "delivered on")) if order.get(k)]
    if dates or order.get("delivery_attempts"): L.append("Dates: " + "; ".join(dates) + (f"; delivery_attempts={order['delivery_attempts']}" if order.get("delivery_attempts") else ""))
    pm = order.get("payment_method")
    if pm in REFUND_TIMING:
        L.append(f"Refund timing for this order's payment method ({pm}) once a refund is initiated: {REFUND_TIMING[pm]} (REF-2)")
    L.append(f"Amounts: subtotal Rs {order.get('subtotal_inr')}, delivery fee Rs {order.get('delivery_fee_inr')}, COD fee Rs {order.get('cod_fee_inr')}, total Rs {order.get('total_inr')}"
             + (f"; coupon {order['coupon_code']} gave Rs {order.get('coupon_discount_inr')} off (min order Rs {order.get('coupon_min_order_inr')})" if order.get("coupon_code") else ""))
    for it in order.get("items", []):
        p = products.get(it["sku"], {})
        extra = f"; fulfilled_by={p.get('fulfilled_by')}; Assured badge={p.get('assured')}; brand warranty={p.get('warranty_months')} months" if p else ""
        L.append(f"Item line {it['line_no']}: {it['title']} (sku {it['sku']}, category {it['return_category']}, qty {it['qty']}, Rs {it['unit_price_inr']} each{extra})")
    if order.get("delivered_on"):
        for it in order.get("items", []):
            cat = it["return_category"]
            days = calc.return_window_days(cat, order["delivered_on"], placed)
            if days is None:
                L.append(f"Return window for '{it['title']}': category {cat} is not returnable (RET-5)"); continue
            dl = calc.return_deadline(cat, order["delivered_on"], placed)
            info["return_deadlines"][it["sku"]] = dl; info["return_windows"][it["sku"]] = days
            L.append(f"Return window for '{it['title']}': {days} calendar days from delivery on {fmt(_d(order['delivered_on']))} -> last return date {fmt(dl)} "
                     f"({'still open' if today <= dl else 'closed'} as of {fmt(today)}); pickup fee for change-of-mind returns Rs {calc.pickup_fee_inr(cat, bool(order.get('is_plus')), False)}")
    elif order.get("status") in ("shipped", "out_for_delivery", "delivery_attempted", "packed", "placed"):
        for it in order.get("items", []):
            days = calc.return_window_days(it["return_category"], today, placed)
            if days:
                info["return_windows"][it["sku"]] = days
                L.append(f"If the customer accepts the delivery, the return window for '{it['title']}' will be {days} calendar days from delivery (category {it['return_category']})")
    if order.get("return_sku"):
        L.append(f"Return: sku {order['return_sku']}, status={order.get('return_status')}, reason={order.get('return_reason')}, requested {order.get('return_requested_on')}, "
                 f"pickup {order.get('return_pickup_on')}, QC passed {order.get('return_qc_passed_on')}, pickup fee deducted Rs {order.get('return_reverse_pickup_fee_inr') or 0}")
    if order.get("return_status") and not order.get("return_refund_initiated_on"):
        L.append("Refund process after a return pickup: the item first goes through a quality check (QC) at the warehouse, and the refund is initiated only after it passes (RET-7)")
    for label, a, b, c, d, e in (("Return", "return_refund_initiated_on", "return_refund_method", "return_refund_amount_inr", "return_rrn", "return_refund_completed_on"),
                                 ("Cancellation", "cancellation_refund_initiated_on", "cancellation_refund_method", "cancellation_refund_amount_inr", "cancellation_rrn", None),
                                 ("Returned-to-origin", "rto_refund_initiated_on", "rto_refund_method", "rto_refund_amount_inr", "rto_rrn", None)):
        line, inf = _refund_line(label, order.get(a), order.get(b), order.get(c), order.get(d), order.get(e) if e else None, H)
        if line:
            L.append(line); info["refunds"].append(inf)
            if inf["overdue"]: info["overdue"].append(inf)
    if order.get("express_late_credit_inr"):
        L.append(f"Express late-delivery credit: Rs {order['express_late_credit_inr']} credited to Wallet on {order.get('express_late_credit_on')}")
    elif order.get("delivery_type") == "express" and order.get("delivered_on") and order.get("promised_date") and _d(order["delivered_on"]) > _d(order["promised_date"]):
        L.append(f"Express order delivered {(_d(order['delivered_on']) - _d(order['promised_date'])).days} day(s) after the promised date; no late credit is recorded")
    if order.get("damage_sku"):
        info["damage_present"] = True
        val = sum(int(it["unit_price_inr"]) * int(it["qty"]) for it in order.get("items", []) if it["sku"] == order["damage_sku"])
        info["damage_high_value"] = val > 10000
        L.append(f"Damage report: sku {order['damage_sku']}, issue={order.get('damage_issue')}, reported {order.get('damage_hours_after_delivery')} hours after delivery, "
                 f"photos attached={order.get('damage_photos_attached')}, status={order.get('damage_status')}; item value Rs {val}"
                 + (" (above Rs 10,000 -> Priority Resolution / human handover, ESC-1.1)" if info["damage_high_value"] else ""))
    L.append(f"Earlier support contacts about this order in the last 7 days: {info['contacts']}")
    if order.get("timeline"): L.append(f"Timeline: {order['timeline']}")
    return L, info

def inline_facts(text: str, topic: str, H, products: dict, today=calc.TODAY):
    """For questions that carry their own data (a date, a payment method, a product) and need no order lookup."""
    L, info = [], {"overdue": False, "high_value_damage": False, "deadline": None, "refund_latest": None, "products": []}
    dates, method = parse_dates(text, today), detect_payment_method(text)
    prods = find_products(text, products, allow_prefix_fuzzy=(topic == "return"))
    info["products"] = prods
    for p in prods:
        eff = p["assured"].strip().lower() == "true" and p["fulfilled_by"] == "ZentroMart"
        category = p["return_category"]
        L.append(f"Product {p['title']} (sku {p['sku']}): category {category}, price Rs {p['price_inr']}, fulfilled_by {p['fulfilled_by']}, "
                 f"Assured badge shown={p['assured']}, ZentroMart Assured warranty actually applies={eff} (MKS-3), brand warranty {p['warranty_months']} months")
        if topic == "warranty":
            brand_months = int(p.get("warranty_months") or 0)
            if eff:
                L.append("POLICY ANSWER: This product has a 12-month ZentroMart Assured warranty from delivery in addition to any brand warranty. Claims are raised in My Orders in the app; ZentroMart arranges pickup and repair or replacement (WAR-2).")
            elif brand_months:
                L.append(f"POLICY ANSWER: The catalog lists a {brand_months}-month brand warranty for this product. Brand warranty claims are made with the manufacturer's service centre (WAR-1).")
            else:
                L.append("POLICY ANSWER: The product catalog does not list a brand warranty period for this product, and ZentroMart Assured warranty does not apply based on the supplied catalog fields. I cannot confirm any additional warranty terms beyond those records (WAR-1, WAR-2).")
        elif topic == "return":
            if category in calc.NOT_RETURNABLE:
                L.append("POLICY ANSWER: This product category is not returnable for change of mind (RET-5). If it arrived damaged, expired, wrong or missing, use the damaged-item process instead.")
            elif category == "fashion":
                L.append("POLICY ANSWER: Clothing, footwear and bags normally have a 15-calendar-day return window from delivery. Orders placed from 1 to 15 October 2026 qualify for a 30-day festive-sale window (RET-2, RET-8). A ₹49 reverse-pickup fee applies to change-of-mind fashion returns unless the customer is a Plus member or the item is damaged, defective or wrong (RET-6).")
            elif category == "electronics":
                L.append("POLICY ANSWER: Electronics and accessories have a 7-calendar-day return window from delivery for deliveries on or after 1 August 2026; deliveries before that date retain the earlier 10-day window. Items must not be physically damaged, and locks must be removed (RET-2, RET-3, RET-4).")
            elif category == "beauty_sealed":
                L.append("POLICY ANSWER: Beauty and personal-care products have a 7-calendar-day return window only if unopened with the seal intact (RET-2, RET-4).")
            else:
                days = calc.RETURN_WINDOWS.get(category)
                if days:
                    L.append(f"POLICY ANSWER: This product category has a {days}-calendar-day return window from delivery (RET-2).")
        elif topic == "damage":
            L.append("POLICY ANSWER: Report a damaged, defective, expired, wrong or missing item within 48 hours of delivery with photos of the item and packaging. After verification, the customer gets a free replacement or full refund including delivery fees, and no reverse-pickup fee applies (DWM-1, DWM-2).")
    if topic == "refund" and method and not dates:
        L.append(f"POLICY ANSWER: {REFUND_TIMING[method]} (REF-2, REF-3). Refund timelines are counted in business days from the date of initiation, except instant Wallet refunds.")
    if topic in ("refund", "other", "payment") and dates and method in calc.REFUND_BUSINESS_DAYS and re.search(r"re[\s-]*fund|रिफ़ंड|रिफंड|paise|पैसे", text, re.I):
        d = dates[0]; lo, hi = calc.refund_eta(d, method, H); info["refund_latest"] = hi
        L.append(f"CALCULATED: a {method} refund initiated on {fmt(d)} is expected between {fmt(lo)} and {fmt(hi)}; the latest date is {fmt(hi)} (business days are Mon-Sat excluding holidays)")
        if calc.refund_overdue(d, method, H, today):
            info["overdue"] = True
            L.append("This refund is more than 3 business days past its maximum timeline -> hand over to a human (ESC-1.5)")
    if topic == "return" and dates and prods:
        d, p = dates[0], prods[0]; cat = p["return_category"]
        if cat == "fashion":
            std, fest = calc.return_deadline(cat, d, "2026-09-01"), calc.return_deadline(cat, d, "2026-10-05")
            L.append(f"CALCULATED: delivered {fmt(d)}; fashion window is 15 days -> {fmt(std)}, or 30 days -> {fmt(fest)} if the order was placed between 1 and 15 October 2026 (RET-8)")
        else:
            dl = calc.return_deadline(cat, d, d); days = calc.return_window_days(cat, d, d)
            if dl: info["deadline"] = dl; info["window_days"] = days; L.append(f"CALCULATED: delivered {fmt(d)}; {cat} window is {days} days -> last return date {fmt(dl)}")
            else: L.append(f"Category {cat} is not returnable (RET-5)")
    if topic == "damage" and prods and int(prods[0]["price_inr"]) > 10000:
        info["high_value_damage"] = True
        L.append(f"The item costs Rs {prods[0]['price_inr']} (above Rs 10,000): damaged/defective high-value items go to a human (ESC-1.1, DWM-3)")
    return L, info
