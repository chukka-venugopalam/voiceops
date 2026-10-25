"""Deterministic rules: business days, refund ETAs, return deadlines. Let code do this, not the LLM."""
import csv, os
from datetime import date, datetime, timedelta
from .config import dataset_dir

TODAY = date(2026, 10, 20)                      # fixed dataset date (KB-00.2)
ELECTRONICS_CUTOFF = date(2026, 8, 1)           # RET-3
FESTIVE_START, FESTIVE_END = date(2026, 10, 1), date(2026, 10, 15)   # RET-8
REFUND_BUSINESS_DAYS = {                        # REF-2: (earliest, latest) business days after initiation
    "upi": (1, 3), "card": (5, 7), "netbanking": (3, 5),
    "wallet": (0, 0), "cod_wallet": (0, 0), "cod_bank": (5, 7),
}
RETURN_WINDOWS = {"electronics": 7, "fashion": 15, "home_kitchen": 10, "beauty_sealed": 7, "books_media": 10}  # RET-2
NOT_RETURNABLE = {"grocery", "innerwear", "digital"}

def to_date(x) -> date:
    if isinstance(x, datetime): return x.date()
    if isinstance(x, date): return x
    return date.fromisoformat(str(x).strip()[:10])

def load_holidays(root=None) -> set:
    root = root or dataset_dir()
    with open(os.path.join(root, "data", "holidays.csv"), encoding="utf-8", newline="") as f:
        return {date.fromisoformat(r["date"]) for r in csv.DictReader(f)}

def is_business_day(d: date, holidays) -> bool:
    return d.weekday() != 6 and d not in holidays        # Mon-Sat, minus holidays (KB-00.3)

def add_business_days(start, n: int, holidays) -> date:
    d, left = to_date(start), n
    while left > 0:
        d += timedelta(days=1)
        if is_business_day(d, holidays): left -= 1
    return d

def business_days_between(start, end, holidays) -> int:
    """Business days in (start, end]."""
    d, end, c = to_date(start), to_date(end), 0
    while d < end:
        d += timedelta(days=1)
        if is_business_day(d, holidays): c += 1
    return c

def refund_eta(initiated_on, method: str, holidays):
    lo, hi = REFUND_BUSINESS_DAYS[method]
    return add_business_days(initiated_on, lo, holidays), add_business_days(initiated_on, hi, holidays)

def refund_overdue(initiated_on, method: str, holidays, today=TODAY, grace: int = 3) -> bool:
    """ESC-1.5: more than 3 business days past the MAXIMUM timeline."""
    _, latest = refund_eta(initiated_on, method, holidays)
    return business_days_between(latest, today, holidays) > grace

def return_window_days(category: str, delivered_on, placed_on):
    if category in NOT_RETURNABLE: return None
    delivered_on, placed_on = to_date(delivered_on), to_date(placed_on)
    if category == "electronics":
        return 10 if delivered_on < ELECTRONICS_CUTOFF else 7
    if category == "fashion":
        return 30 if FESTIVE_START <= placed_on <= FESTIVE_END else 15
    return RETURN_WINDOWS[category]

def return_deadline(category: str, delivered_on, placed_on):
    days = return_window_days(category, delivered_on, placed_on)
    return None if days is None else to_date(delivered_on) + timedelta(days=days)

def return_still_open(category, delivered_on, placed_on, today=TODAY) -> bool:
    dl = return_deadline(category, delivered_on, placed_on)
    return dl is not None and today <= dl

def pickup_fee_inr(category: str, is_plus: bool, damaged_or_wrong: bool) -> int:
    """RET-6: Rs 49 only for change-of-mind fashion returns; waived for Plus and damaged/wrong."""
    return 49 if (category == "fashion" and not is_plus and not damaged_or_wrong) else 0

def standard_delivery_fee_inr(order_value_inr: int) -> int:
    return 0 if order_value_inr >= 499 else 40            # DEL-2

def cod_allowed(order_value_inr: int, pincode_cod_available: bool) -> bool:
    return pincode_cod_available and order_value_inr <= 10000   # PAY-2
