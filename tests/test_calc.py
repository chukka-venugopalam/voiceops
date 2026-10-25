from datetime import date
from voiceops import calc
from conftest import needs_data
pytestmark = needs_data

def test_refund_dates_match_practice_keys():
    H = calc.load_holidays()
    for method, start, gold in [("card", "2026-10-17", "2026-10-27"), ("upi", "2026-10-12", "2026-10-15"), ("netbanking", "2026-10-14", "2026-10-21"),
                                ("card", "2026-10-12", "2026-10-21"), ("netbanking", "2026-10-17", "2026-10-24"), ("netbanking", "2026-10-12", "2026-10-17"),
                                ("upi", "2026-10-17", "2026-10-22")]:
        assert calc.refund_eta(start, method, H)[1] == date.fromisoformat(gold)

def test_return_windows_and_fees():
    assert calc.return_deadline("fashion", "2026-10-12", "2026-09-30") == date(2026, 10, 27)
    assert calc.return_window_days("electronics", "2026-07-20", "2026-07-15") == 10
    assert calc.return_window_days("electronics", "2026-10-18", "2026-10-10") == 7
    assert calc.return_window_days("fashion", "2026-10-12", "2026-10-05") == 30
    assert calc.return_window_days("innerwear", "2026-10-12", "2026-10-05") is None
    assert calc.pickup_fee_inr("fashion", False, False) == 49 and calc.pickup_fee_inr("fashion", True, False) == 0
    assert calc.cod_allowed(12000, True) is False and calc.standard_delivery_fee_inr(350) == 40

def test_overdue_rule():
    H = calc.load_holidays()
    assert calc.refund_overdue("2026-07-20", "upi", H) is True
    assert calc.refund_overdue("2026-10-17", "card", H) is False
