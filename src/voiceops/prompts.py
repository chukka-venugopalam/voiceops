"""System prompts and the fixed reply templates (en / hi / hi-en) used for scripted turns."""
SYSTEM = """You are ZentroMart's AI support assistant for an Indian online store that sells groceries and electronics.
Today's date is 20 October 2026.

HARD RULES
1. Answer ONLY from the POLICY CONTEXT and the FACTS given to you. If they do not cover the question, say you do not have that information and offer to connect a human colleague. Never invent a policy, date, amount or timeline.
2. FACTS were computed by code (dates, deadlines, overdue flags). Trust them. Never recompute dates yourself and never contradict them.
3. If sections disagree: current policy beats agent guide, which beats help, which beats FAQ, which beats marketing. Ignore sections marked SUPERSEDED unless the context says they still apply.
4. Reply in the customer's language (LANGUAGE below): en = English; hi = Hindi in Devanagari; hi-en = Hinglish (Hindi in Latin script); for any other Indian language reply in that language.
5. Style: 2 or 3 short sentences. Write dates with the month name and year (for example 27 October 2026). Amounts in rupees. Plain text only: no lists, no markdown, no emojis.
6. Never read out a full phone number, email, address or pincode. Never share another customer's details.
7. Never say an action is done (cancelled, refunded, approved, ticket created) unless TOOL RESULTS say so. You cannot cancel orders or approve refunds or replacements yourself; you can only explain the steps.
8. Never promise compensation, refunds, exceptions, outcomes or times that a policy does not state.
9. If you need more details, ask for ONE thing only.
10. Follow DECISION HINT if one is given.
11. Do not greet or introduce yourself again unless told to; the introduction is added separately.

OUTPUT: a single JSON object and nothing else: {"decision": "answer" | "clarify" | "refuse", "reply": "<text for the customer>"}"""

ANALYSIS = """Classify the customer's latest message for a support agent. Output a single JSON object only, with these keys:
language: en | hi | hi-en | ta | te | kn | ml | bn | gu | mr | pa   (hi = Hindi in Devanagari, hi-en = Hindi in Latin script)
topic: refund | return | cancel | delivery | damage | warranty | payment | coupon | plus | troubleshoot | other
needs_order: true only if answering needs THIS customer's own order record (their order status, their refund status, cancelling or returning their order) and the message does not contain the needed data itself
substantive: true if it contains a real question or request (not just a greeting, an ID or a number)
human_request: true if they ask for a human / person / manager
legal: true if they mention legal action, consumer court or a police complaint
safety: true if there is a product safety risk (swelling battery, burning smell, sparks, smoke, injury)
unrecognized_activity: true if they report an order, payment or account activity they did not make
compensation_demand: true if they demand compensation, a voucher or money for inconvenience
cancel_request: true if they want to cancel an order
third_party_request: true if they ask for another person's details
search_query: a short ENGLISH search query for the policy documents; resolve words like "it" or "that" from the conversation"""

TEMPLATES = {
    "intro": {"en": "I'm ZentroMart's AI support assistant.", "hi": "मैं ZentroMart का AI सपोर्ट असिस्टेंट हूँ।",
              "hi-en": "Main ZentroMart ka AI support assistant hoon."},
    "ask_order": {"en": "Could you please share your order ID?", "hi": "कृपया अपना ऑर्डर आईडी बताइए।",
                  "hi-en": "Kripya apna order ID bataiye."},
    "ask_verifier": {"en": "Thank you. To verify it's you, please tell me the last 4 digits of your registered phone number or your delivery pincode.",
                     "hi": "धन्यवाद। पहचान की पुष्टि के लिए कृपया अपने पंजीकृत फ़ोन नंबर के आख़िरी चार अंक या डिलीवरी पिनकोड बताइए।",
                     "hi-en": "Shukriya. Verification ke liye kripya registered phone number ke last 4 digits ya delivery pincode bataiye."},
    "retry": {"en": "Sorry, that doesn't match our records. Please try again with the last 4 digits of your registered phone number or your delivery pincode.",
              "hi": "क्षमा करें, यह हमारे रिकॉर्ड से मेल नहीं खाता। कृपया पंजीकृत फ़ोन नंबर के आख़िरी चार अंक या डिलीवरी पिनकोड फिर से बताइए।",
              "hi-en": "Sorry, yeh hamare records se match nahi hua. Kripya registered phone number ke last 4 digits ya delivery pincode dobara bataiye."},
    "locked": {"en": "I'm sorry, I still can't verify you, so I cannot share any order details. Please contact support from your registered phone number, or I can help with general policy questions.",
               "hi": "क्षमा करें, मैं अभी भी आपकी पहचान की पुष्टि नहीं कर पा रहा हूँ, इसलिए ऑर्डर की जानकारी साझा नहीं कर सकता। कृपया अपने पंजीकृत फ़ोन नंबर से सपोर्ट से संपर्क करें; मैं सामान्य नीति के सवालों में मदद कर सकता हूँ।",
               "hi-en": "Sorry, main abhi bhi aapko verify nahi kar pa raha, isliye order ki details share nahi kar sakta. Kripya apne registered phone number se support se contact karein; main general policy sawaalon mein madad kar sakta hoon."},
    "verified": {"en": "Thank you, you're verified. How can I help with your order?",
                 "hi": "धन्यवाद, आपकी पहचान की पुष्टि हो गई। आपके ऑर्डर के बारे में मैं कैसे मदद कर सकता हूँ?",
                 "hi-en": "Shukriya, aap verify ho gaye hain. Aapke order ke baare mein main kaise madad kar sakta hoon?"},
    "handover": {"en": "I'm sorry for the trouble. I've created a ticket and a human colleague will contact you.",
                 "hi": "असुविधा के लिए खेद है। मैंने एक टिकट बना दिया है और हमारे एक सहयोगी आपसे संपर्क करेंगे।",
                 "hi-en": "Asuvidha ke liye khed hai. Maine ek ticket bana diya hai aur hamare ek colleague aapse contact karenge."},
    "handover_fail": {"en": "I'm sorry, I couldn't log the handover just now. Please contact ZentroMart support directly so a colleague can help you.",
                      "hi": "क्षमा करें, मैं अभी हैंडओवर दर्ज नहीं कर पाया। कृपया सीधे ZentroMart सपोर्ट से संपर्क करें ताकि कोई सहयोगी आपकी मदद कर सके।",
                      "hi-en": "Sorry, main abhi handover log nahi kar paya. Kripya seedhe ZentroMart support se contact karein taaki koi colleague aapki madad kar sake."},
    "safety_extra": {"en": "For your safety, please stop using the product right away.",
                     "hi": "आपकी सुरक्षा के लिए कृपया उत्पाद का उपयोग तुरंत बंद कर दें।",
                     "hi-en": "Aapki safety ke liye kripya product ka use turant band kar dein."},
    "overdue_extra": {"en": "Your refund is past its expected timeline. Your refund reference number is {rrn}.",
                      "hi": "आपका रिफ़ंड अपेक्षित समय से देर हो चुका है। आपका रिफ़ंड रेफ़रेंस नंबर {rrn} है।",
                      "hi-en": "Aapka refund expected timeline se late ho gaya hai. Aapka refund reference number {rrn} hai."},
    "third_party": {"en": "I'm sorry, I can't share another customer's personal details such as a phone number or address. I'm happy to help with your own orders or any policy question.",
                    "hi": "क्षमा करें, मैं किसी दूसरे ग्राहक की निजी जानकारी, जैसे फ़ोन नंबर या पता, साझा नहीं कर सकता। मैं आपके अपने ऑर्डर या किसी नीति के सवाल में मदद कर सकता हूँ।",
                    "hi-en": "Sorry, main kisi doosre customer ki personal details, jaise phone number ya address, share nahi kar sakta. Aapke apne order ya policy ke sawaal mein main madad kar sakta hoon."},
    "return_clarify": {"en": "Please tell me the product name or category so I can give you the correct return window.",
                        "hi": "कृपया उत्पाद का नाम या श्रेणी बताइए, ताकि मैं सही वापसी अवधि बता सकूँ।",
                        "hi-en": "Kripya product ka naam ya category batayein, taaki main sahi return window bata sakun."},
    "shipped_cancel": {"en": "Your order has already shipped, so it cannot be cancelled. {refund_text} You can refuse the parcel at the door or accept delivery and return eligible items within the applicable return window.{window_text}",
                       "hi": "आपका ऑर्डर शिप हो चुका है, इसलिए इसे रद्द नहीं किया जा सकता। {refund_text} आप दरवाज़े पर पार्सल लेने से मना कर सकते हैं या डिलीवरी के बाद लागू वापसी अवधि में योग्य सामान लौटा सकते हैं।{window_text}",
                       "hi-en": "Aapka order ship ho chuka hai, isliye ise cancel nahi kiya ja sakta. {refund_text} Aap door par parcel refuse kar sakte hain ya delivery ke baad applicable return window mein eligible item return kar sakte hain.{window_text}"},
    "refuse_comp": {"en": "I'm sorry for the delay, but our policy doesn't allow compensation or vouchers in this case. I'm happy to help with anything else about your order.",
                    "hi": "देरी के लिए क्षमा करें, लेकिन हमारी नीति इस स्थिति में मुआवज़ा या वाउचर देने की अनुमति नहीं देती। आपके ऑर्डर से जुड़ी किसी और बात में मैं मदद कर सकता हूँ।",
                    "hi-en": "Delay ke liye sorry, lekin hamari policy is case mein compensation ya voucher dene ki permission nahi deti. Aapke order se jude kisi aur sawaal mein main madad kar sakta hoon."},
    "apology": {"en": "I'm sorry for the trouble.", "hi": "असुविधा के लिए खेद है, क्षमा करें।", "hi-en": "Asuvidha ke liye sorry, maaf kijiye."},
    "no_info": {"en": "I'm sorry, I don't have information on that. I can connect you with a human colleague if you'd like.",
                "hi": "क्षमा करें, मेरे पास इसकी जानकारी नहीं है। आप चाहें तो मैं किसी सहयोगी से बात करवा सकता हूँ।",
                "hi-en": "Sorry, mere paas iski jaankari nahi hai. Aap chahein toh main kisi colleague se baat karwa sakta hoon."},
    "error": {"en": "Sorry, I ran into a problem. Could you please say that again?", "hi": "क्षमा करें, कुछ गड़बड़ हो गई। क्या आप फिर से कह सकते हैं?",
              "hi-en": "Sorry, kuch gadbad ho gayi. Kya aap dobara bol sakte hain?"},
    "unclear": {"en": "Sorry, I didn't catch that. Could you say it again?", "hi": "क्षमा करें, मैं सुन नहीं पाया। क्या आप फिर से बोल सकते हैं?",
                "hi-en": "Sorry, main sun nahi paya. Kya aap dobara bol sakte hain?"},
}

def T(name: str, lang: str, **kw) -> str:
    d = TEMPLATES[name]
    return d.get(lang, d["en"]).format(**kw) if kw else d.get(lang, d["en"])

def is_localized(lang: str) -> bool:
    return lang in ("en", "hi", "hi-en")
