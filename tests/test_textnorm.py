from datetime import date
from voiceops.textnorm import extract_ids, detect_language, parse_dates, spoken_to_digits, detect_payment_method
from voiceops.speech_text import speechify

def test_ids_grouped_spoken_and_devanagari():
    assert extract_ids("Order number hai 1859 9253.")["order_id"] == "18599253"
    assert extract_ids("my order is one eight five nine nine two five three")["order_id"] == "18599253"
    assert extract_ids("Last four digits 4265 hain.")["verifier"] == "4265"
    assert extract_ids("आख़िरी चार अंक १२६६ हैं।")["verifier"] == "1266"
    assert extract_ids("pincode 600 042")["verifier"] == "600042"
    assert extract_ids("last four double four six five")["verifier"] == "4465"
    assert extract_ids("it was on 17 October 2026")["verifier"] is None      # a year is not a phone suffix

def test_language():
    assert detect_language("Mera refund kahan hai?") == "hi-en"
    assert detect_language("क्या मैं किसी इंसान से बात कर सकती हूँ?") == "hi"
    assert detect_language("Can I speak to a person?") == "en"
    assert detect_language("4265", "hi") == "hi"

def test_dates_and_methods():
    assert parse_dates("refund 17 October two zero two six") == [date(2026, 10, 17)]
    assert parse_dates("delivered parso") == [date(2026, 10, 18)]
    assert parse_dates("17 अक्टूबर 2026") == [date(2026, 10, 17)]
    assert detect_payment_method("refund to my u p i") == "upi"
    assert spoken_to_digits("two zero two six") == "2026"

def test_speechify():
    s = speechify("Refund of ₹10,000 for order 18599253 by 27 October 2026. Fee ₹49.")
    assert "ten thousand rupees" in s and "1859 9253" in s and "27th of October 2026" in s and "forty-nine rupees" in s
