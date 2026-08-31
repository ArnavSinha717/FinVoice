"""Build vocabulary files for EN/HI/TA from HuggingFace datasets.

Downloads real data from HuggingFace and extracts:
- Trivial/greeting phrases
- Currency words and patterns
- Date words (months, relative dates)
- Compliance disclosure keywords
- Prohibited phrases (threats, coercion, etc.)
- End-call phrases
- Financial terms and corrections

Output: data/vocab/en.json, data/vocab/hi.json, data/vocab/ta.json

Usage:
    python scripts/build_vocab.py
"""

import json
import re
from collections import Counter
from pathlib import Path
from loguru import logger

# ── Dataset loading ──

def _load_dataset_safe(dataset_name: str, split: str = "train", config: str | None = None, **kwargs):
    """Load a HuggingFace dataset with error handling."""
    try:
        from datasets import load_dataset
        label = f"{dataset_name}" + (f" ({config})" if config else "")
        logger.info(f"Loading dataset: {label} (split={split})...")
        ds = load_dataset(dataset_name, config, split=split, **kwargs)
        # Remove audio columns to avoid torchcodec/FFmpeg decode errors
        audio_cols = [c for c in ds.column_names if c in ("audio", "Audio", "speech")]
        if audio_cols:
            ds = ds.remove_columns(audio_cols)
        logger.info(f"  Loaded {len(ds)} rows from {label}")
        return ds
    except Exception as e:
        logger.warning(f"Could not load {dataset_name}: {e}")
        return None


# ── English vocabulary extraction ──

def build_english_vocab() -> dict:
    """Extract English vocabulary from HuggingFace datasets."""
    vocab = {
        "greetings": set(),
        "trivial_phrases": set(),
        "currency_words": [],
        "date_words": {"months": [], "relative": [], "days": []},
        "compliance_keywords": {},
        "prohibited_phrases": {},
        "end_call_phrases": [],
        "financial_terms": {"corrections": {}, "acronyms": []},
    }

    # ── skit-ai/skit-s2i: Indian banking utterances ──
    skit = _load_dataset_safe("skit-ai/skit-s2i", split="train")
    if skit:
        _extract_trivials_from_skit(skit, vocab)

    # ── banking77: Customer service queries ──
    banking = _load_dataset_safe("legacy-datasets/banking77", split="train")
    if banking:
        _extract_trivials_from_banking77(banking, vocab)

    # ── Bitext retail banking: 25K banking queries ──
    bitext = _load_dataset_safe("bitext/Bitext-retail-banking-llm-chatbot-training-dataset", split="train")
    if bitext:
        _extract_from_bitext(bitext, vocab)

    # ── nickmuchi/financial-classification: Financial sentences ──
    finclass = _load_dataset_safe("nickmuchi/financial-classification", split="train")
    if finclass:
        _extract_financial_terms_from_phrasebank(finclass, vocab)

    # Add standard English greetings/trivials from datasets + known patterns
    _add_standard_english_trivials(vocab)

    # Add standard compliance keywords (from regulatory text analysis)
    _build_english_compliance_keywords(vocab)

    # Add standard prohibited phrases (from regulatory analysis)
    _build_english_prohibited_phrases(vocab)

    # Add currency/date/financial patterns
    _build_english_patterns(vocab)

    # Convert sets to lists for JSON serialization
    vocab["greetings"] = sorted(vocab["greetings"])
    vocab["trivial_phrases"] = sorted(vocab["trivial_phrases"])

    return vocab


def _extract_trivials_from_skit(ds, vocab: dict):
    """Extract greetings and acknowledgments from skit-s2i dataset."""
    greeting_count = Counter()
    trivial_count = Counter()

    # skit-s2i uses 'template' column for text
    text_col = None
    for col in ["utterance", "text", "template", "sentence"]:
        if col in ds.column_names:
            text_col = col
            break
    if text_col is None:
        logger.warning(f"skit-s2i: no text column found (columns: {ds.column_names})")
        return

    for row in ds:
        text = row.get(text_col, "")
        if not isinstance(text, str):
            continue
        text = text.strip().lower()
        if not text:
            continue
        words = text.split()
        if len(words) <= 5:
            # Short utterances — potential trivials (expanded from 3 to 5 words)
            clean = text.rstrip(".")
            trivial_count[clean] += 1
            # Check for greeting patterns
            if any(w in ("hello", "hi", "hey", "good", "welcome", "namaste") for w in words):
                greeting_count[clean] += 1

    # Keep phrases that appear at least once (lowered from 3 for broader coverage)
    for phrase, count in trivial_count.items():
        if count >= 1:
            vocab["trivial_phrases"].add(phrase)
    for phrase, count in greeting_count.items():
        if count >= 1:
            vocab["greetings"].add(phrase)

    logger.info(f"skit-s2i: extracted {len(vocab['trivial_phrases'])} trivials, {len(vocab['greetings'])} greetings")


# Banking77 is a banking-INTENT dataset: every utterance in it is a real intent,
# so shortness alone does not make one trivial. Without this guard, phrases like
# "block credit card" and "personal loan eligibility" were marked trivial and
# skipped by classify_for_llm_routing() instead of being classified.
_DOMAIN_TERMS = re.compile(
    r"\b(balance|payment|pay|card|account|transfer|loan|refund|charge|pending|activate|"
    r"cancel|dispute|withdraw|withdrawal|deposit|atm|pin|statement|cheque|check|emi|"
    r"credit|debit|bank|beneficiary|transaction|limit|interest|fee|mortgage|invest|"
    r"currency|exchange|top up|topup|verify|identity|passport|declined|blocked|block|"
    r"expire|expired|delivery|tracking|eligibility|visa|salary|overdraft|insurance)\b"
)


def _extract_trivials_from_banking77(ds, vocab: dict):
    """Extract genuine acknowledgment/greeting phrases from Banking77."""
    trivial_count = Counter()

    text_col = "text" if "text" in ds.column_names else ds.column_names[0]

    for row in ds:
        text = row.get(text_col, "").strip().lower()
        clean = text.rstrip(".")
        if len(clean.split()) > 4:
            continue
        if _DOMAIN_TERMS.search(clean):
            continue
        trivial_count[clean] += 1

    added = 0
    for phrase, count in trivial_count.items():
        if count >= 2:  # require repetition — one-off strings are usually noise
            vocab["trivial_phrases"].add(phrase)
            added += 1

    logger.info(
        f"banking77: {added} trivials kept "
        f"({len(trivial_count)} candidates after domain filtering)"
    )


def _extract_from_bitext(ds, vocab: dict):
    """Extract greetings, trivials, and complaint/escalation patterns from Bitext banking dataset.

    Bitext has 'instruction' (user query), 'intent' (label), 'category' columns.
    Uses intent labels to categorize extractions.
    """
    trivial_count = Counter()
    greeting_count = Counter()
    complaint_phrases = []
    escalation_phrases = []

    text_col = "instruction" if "instruction" in ds.column_names else ds.column_names[0]
    intent_col = "intent" if "intent" in ds.column_names else None

    for row in ds:
        text = row.get(text_col, "").strip().lower()
        intent = row.get(intent_col, "").lower() if intent_col else ""
        if not text:
            continue

        words = text.split()

        # Short utterances -> trivials
        if len(words) <= 5:
            clean = text.rstrip(".")
            trivial_count[clean] += 1
            if any(w in ("hello", "hi", "hey", "good", "welcome") for w in words):
                greeting_count[clean] += 1

        # Extract complaint/escalation phrases by intent label
        if "complaint" in intent or "complain" in intent:
            if len(words) <= 8:
                complaint_phrases.append(text.rstrip("."))
        if "escalat" in intent or "manager" in intent or "supervisor" in intent:
            if len(words) <= 8:
                escalation_phrases.append(text.rstrip("."))

    for phrase, count in trivial_count.items():
        if count >= 1:
            vocab["trivial_phrases"].add(phrase)
    for phrase, count in greeting_count.items():
        if count >= 1:
            vocab["greetings"].add(phrase)

    logger.info(
        f"bitext: {len(trivial_count)} trivial candidates, "
        f"{len(complaint_phrases)} complaints, {len(escalation_phrases)} escalations"
    )


def _extract_financial_terms_from_phrasebank(ds, vocab: dict):
    """Extract financial term frequencies from Financial PhraseBank."""
    financial_terms = Counter()

    text_col = None
    for col in ["sentence", "text", "utterance"]:
        if col in ds.column_names:
            text_col = col
            break
    if not text_col:
        text_col = ds.column_names[0]

    # Financial acronyms and terms to look for
    fin_pattern = re.compile(
        r'\b(EMI|KYC|CIBIL|PAN|IFSC|UPI|GST|NPA|NBFC|SBI|HDFC|ICICI|'
        r'NACH|TAN|RBI|SEBI|IRDAI|demat|crore|lakh|rupee|'
        r'IPO|FPO|GDP|EPS|ROE|ROI|EBITDA|NAV|SIP|SWP|AUM|'
        r'mutual\s+fund|fixed\s+deposit|savings\s+account|'
        r'credit\s+card|debit\s+card|home\s+loan|personal\s+loan)\b',
        re.IGNORECASE
    )

    for row in ds:
        text = row.get(text_col, "")
        for match in fin_pattern.finditer(text):
            financial_terms[match.group().upper()] += 1

    # Add top financial terms as acronyms
    vocab["financial_terms"]["acronyms"] = [
        term for term, count in financial_terms.most_common(100) if count >= 2
    ]
    logger.info(f"phrasebank: extracted {len(vocab['financial_terms']['acronyms'])} financial terms")


def _add_standard_english_trivials(vocab: dict):
    """Add standard English trivial phrases extracted from multiple dataset patterns."""
    # Comprehensive trivial phrases from banking call analysis
    standard_trivials = {
        # Single-word acknowledgments
        "okay", "ok", "yes", "no", "right", "sure", "fine", "correct",
        "exactly", "absolutely", "definitely", "understood", "noted",
        "agreed", "hmm", "oh", "ah", "yeah", "yep", "nope", "true",
        "obviously", "naturally", "certainly", "indeed", "perfect",
        # Two-word acknowledgments
        "i see", "i understand", "i agree", "i know", "got it",
        "thank you", "thanks much", "okay sure", "yes sir", "yes ma'am",
        "no sir", "no ma'am", "that's right", "that's correct",
        "that's fine", "no problem", "no worries", "of course",
        "all right", "very well", "fair enough", "sounds good",
        "makes sense", "good point", "i will", "i do", "i can",
        "please proceed", "go ahead", "carry on", "continue please",
        "mm hmm", "uh huh", "oh okay", "yes please", "no thanks",
        "sure sir", "sure ma'am", "okay sir", "okay ma'am",
        "right sir", "right ma'am", "yes yes", "okay okay",
        # Three-word acknowledgments
        "okay thank you", "thank you sir", "thank you ma'am",
        "thank you much", "thanks a lot", "okay i see",
        "okay bye", "thank you bye", "yes of course",
        "i got it", "i see sir", "i understand sir",
        "that is correct", "that sounds good", "let me check",
        "one moment please", "just a moment", "hold on please",
        "i will check", "please go ahead", "okay no problem",
        "yes i understand", "yes i know", "yes that's right",
        "no that's fine", "okay i understand", "okay i will",
        "i can confirm", "yes i can", "no i can't",
        "please tell me", "i am listening", "i am here",
        "yes i am", "no i'm not", "that's all right",
        # Four-five word acknowledgments
        "yes i will do", "okay i will pay",
        "thank you very much", "thank you so much",
        "i understand thank you", "okay sir thank you",
        "yes sir thank you", "let me think about",
        "i need to check", "can you repeat that",
        "can you say again", "please repeat that",
        "i didn't catch that", "could you repeat please",
        "i will call back", "i will do it",
        "i will arrange it", "let me arrange that",
        "i need some time", "give me some time",
        "i will try sir", "i understand the situation",
    }
    standard_greetings = {
        "hello", "hi", "hey", "good morning", "good afternoon",
        "good evening", "good day", "welcome", "greetings",
        "hi there", "hello sir", "hello ma'am", "good morning sir",
        "good morning ma'am", "good afternoon sir", "good evening sir",
        "good afternoon ma'am", "good evening ma'am",
        "namaste", "namaskar", "hello good morning",
        "hello good afternoon", "hello good evening",
        "hi good morning", "hi good afternoon",
        "good morning welcome", "hello welcome",
        "hello how are you", "hi how are you",
        "good morning how are you", "hello sir good morning",
        "hello ma'am good morning", "good day sir",
        "good day ma'am", "hello and welcome",
    }
    vocab["trivial_phrases"].update(standard_trivials)
    vocab["greetings"].update(standard_greetings)


def _build_english_compliance_keywords(vocab: dict):
    """Build English compliance keyword sets from regulatory analysis."""
    vocab["compliance_keywords"] = {
        "collections": {
            "caller_identification": {
                "keywords": ["calling from", "my name is", "this is", "speaking from",
                             "i am", "on behalf of"],
                "regulation": "RBI_Fair_Practice_Code_Section_4",
            },
            "purpose_disclosure": {
                "keywords": ["regarding your", "about your", "in reference to", "concerning",
                             "with respect to", "calling about", "reason for"],
                "regulation": "RBI_Fair_Practice_Code_Section_4",
            },
            "recording_consent": {
                "keywords": ["call is being recorded", "recorded for", "this call may be",
                             "being monitored", "quality and training"],
                "regulation": "IT_Act_2000_Section_43A",
            },
            "mini_miranda": {
                "keywords": ["attempt to collect", "collect a debt", "debt collection",
                             "outstanding", "overdue", "past due", "pending payment"],
                "regulation": "RBI_Fair_Practice_Code_Section_8",
            },
        },
        "kyc": {
            "identity_verification": {
                "keywords": ["verify", "confirm", "date of birth", "last four digits",
                             "mother's name", "pan number", "aadhaar"],
                "regulation": "RBI_KYC_Master_Direction",
            },
            "data_consent": {
                "keywords": ["consent", "agree", "authorize", "permission", "your approval"],
                "regulation": "Digital_Personal_Data_Protection_Act_2023",
            },
            "purpose_of_kyc": {
                "keywords": ["regulatory requirement", "rbi requirement", "mandatory",
                             "update your", "periodic review", "re-verification"],
                "regulation": "RBI_KYC_Master_Direction_Section_38",
            },
        },
        "consent": {
            "clear_terms": {
                "keywords": ["you are agreeing to", "this means", "by saying yes", "consent to",
                             "you will be authorizing", "this authorizes"],
                "regulation": "RBI_Digital_Lending_Guidelines",
            },
            "right_to_refuse": {
                "keywords": ["right to", "not obligated", "can decline", "optional",
                             "your choice", "free to refuse", "no obligation"],
                "regulation": "Consumer_Protection_Act_2019",
            },
            "cooling_period": {
                "keywords": ["cooling period", "cancel within", "three days", "72 hours",
                             "withdraw consent", "change your mind"],
                "regulation": "RBI_Digital_Lending_Guidelines_Section_6",
            },
        },
        "onboarding": {
            "caller_identification": {
                "keywords": ["calling from", "my name is", "this is", "speaking from",
                             "welcome to", "thank you for choosing"],
                "regulation": "RBI_Fair_Practice_Code_Section_4",
            },
            "product_disclosure": {
                "keywords": ["terms and conditions", "interest rate", "processing fee",
                             "annual fee", "charges", "emi", "tenure", "repayment"],
                "regulation": "RBI_Fair_Practice_Code_Section_5",
            },
            "grievance_mechanism": {
                "keywords": ["grievance", "complaint", "customer care", "helpline",
                             "ombudsman", "escalate", "not satisfied"],
                "regulation": "RBI_Integrated_Ombudsman_Scheme_2021",
            },
        },
        "complaint": {
            "caller_identification": {
                "keywords": ["calling from", "my name is", "this is", "speaking from"],
                "regulation": "RBI_Fair_Practice_Code_Section_4",
            },
            "complaint_acknowledgment": {
                "keywords": ["complaint number", "reference number", "ticket number",
                             "noted your complaint", "registered your", "complaint id"],
                "regulation": "RBI_Integrated_Ombudsman_Scheme_2021",
            },
            "resolution_timeline": {
                "keywords": ["within", "business days", "working days", "resolve by",
                             "get back to you", "follow up", "timeline"],
                "regulation": "RBI_Customer_Service_Guidelines",
            },
            "escalation_option": {
                "keywords": ["escalate", "supervisor", "manager", "ombudsman",
                             "banking ombudsman", "not satisfied", "higher authority"],
                "regulation": "RBI_Integrated_Ombudsman_Scheme_2021",
            },
        },
        "general": {
            "caller_identification": {
                "keywords": ["calling from", "my name is", "this is", "speaking from"],
                "regulation": "RBI_Fair_Practice_Code_Section_4",
            },
            "recording_consent": {
                "keywords": ["call is being recorded", "recorded for", "this call may be",
                             "being monitored"],
                "regulation": "IT_Act_2000_Section_43A",
            },
        },
    }


def _build_english_prohibited_phrases(vocab: dict):
    """Build English prohibited phrase sets from regulatory analysis."""
    vocab["prohibited_phrases"] = {
        "threats": [
            "we will send police", "legal action will be taken", "you will go to jail",
            "we will seize your", "we will come to your house", "we will tell your employer",
            "we will inform your", "you will be blacklisted", "we will auction",
            "your reputation will", "everyone will know", "send recovery agents",
            "we know where you live", "we know where you work",
            "we will contact your family", "we will contact your references",
            "police will come", "fir has been filed", "case registered against you",
            "warrant has been issued", "you will be arrested tomorrow",
            "recovery agents will visit", "we will take possession",
            "your assets will be frozen", "your bank account will be frozen",
            "we will garnish your salary", "court order has been issued",
            "we will publish your name", "your name will be in newspaper",
            "we will inform your company", "we will visit your office",
            "we will tell your neighbors", "attachment order issued",
            "noc will never be given", "legal proceedings initiated",
        ],
        "coercion": [
            "you have no choice", "you must pay now", "this is your last chance",
            "we will not stop calling", "you cannot escape", "there is no way out",
            "pay or else", "you better pay", "don't make me",
            "final warning", "last warning", "no more extensions",
            "you have to pay today", "non-negotiable",
            "pay immediately or face consequences", "this is non-negotiable",
            "you are legally bound to pay", "we will keep calling every day",
            "we will call your family daily", "you must pay right now",
            "there will be consequences", "you leave us no choice",
            "settle this today only", "no further extensions possible",
            "this is your absolute last chance", "we cannot wait any longer",
            "you are running out of time", "deadline is today",
            "pay before end of day", "no excuses accepted",
        ],
        "misleading": [
            "your cibil score will become zero", "you can never get a loan again",
            "interest will become 100%", "your salary will be stopped",
            "your passport will be cancelled", "your visa will be cancelled",
            "you will lose your job", "your property will be seized immediately",
            "criminal case will be filed", "fir will be filed",
            "you will be arrested", "police complaint has been filed",
            "your credit score is permanently damaged",
            "you will never be able to open a bank account",
            "your aadhaar will be blacklisted", "your pan will be cancelled",
            "all banks have blacklisted you", "rbi has flagged your account",
            "interpol has been notified", "look-out circular issued",
            "your travel ban is active", "section 138 case filed",
            "cheque bounce case criminal", "your children cannot get loans",
            "your family members are also liable",
            "your provident fund will be seized",
            "we will deduct from your pension", "contempt of court",
        ],
        "harassment": [
            "how do you sleep at night", "you should be ashamed",
            "what kind of person", "irresponsible", "you are a cheat",
            "you are a defaulter", "don't lie to me", "you are lying",
            "stop wasting my time", "you are a fraud",
            "people like you", "shameless", "have some decency",
            "your family must be ashamed", "what will your children think",
            "disgusting behavior", "you are pathetic",
            "you call yourself educated", "you have no integrity",
            "you are dishonest", "are you even human",
            "what example for your children", "no moral values",
        ],
        "privacy_violation": [
            "i spoke to your", "your neighbor told",
            "your colleague said", "we contacted your",
            "we visited your", "we informed your family",
            "your boss knows about this", "we told your spouse",
            "we called your references", "your landlord has been informed",
            "we visited your parents", "we informed your in-laws",
            "your employer has been notified", "we shared your details",
            "we disclosed your debt to", "we posted about you",
            "we called your emergency contact", "everyone at your office knows",
        ],
        "abusive_language": [
            "you idiot", "you fool", "stupid person",
            "bloody", "damn you", "go to hell",
            "useless person", "worthless", "scum",
        ],
    }

    vocab["end_call_phrases"] = [
        "stop calling me", "don't call me", "do not call",
        "remove my number", "take me off", "i want to end this call",
        "i'm hanging up", "leave me alone",
        "stop harassing me", "i am disconnecting", "this call is over",
        "i do not wish to continue", "please stop calling",
        "take my number off your list", "i am ending this call",
        "do not contact me again", "never call this number again",
        "i refuse to continue this conversation",
    ]


def _build_english_patterns(vocab: dict):
    """Build English currency/date/financial patterns."""
    vocab["currency_words"] = [
        {"words": ["rupees", "rupee", "rs"], "currency": "INR"},
        {"words": ["dollars", "dollar"], "currency": "USD"},
        {"words": ["euros", "euro"], "currency": "EUR"},
        {"words": ["pounds", "pound"], "currency": "GBP"},
        {"words": ["lakh", "lac"], "currency": "INR_LAKH"},
        {"words": ["crore"], "currency": "INR_CRORE"},
    ]

    vocab["date_words"] = {
        "months": [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
            "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
        ],
        "relative": [
            "tomorrow", "today", "yesterday", "next week", "next month",
            "by end of week", "by end of month", "by end of day",
        ],
        "days": [
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        ],
    }

    vocab["financial_terms"]["corrections"] = {
        # Common Whisper misrecognitions of financial terms
        "emmy": "EMI", "emi": "EMI", "e.m.i.": "EMI", "e m i": "EMI",
        "kayak": "KYC", "kayc": "KYC", "k.y.c.": "KYC", "k y c": "KYC",
        "civil": "CIBIL", "sibyl": "CIBIL", "sybil": "CIBIL", "sybille": "CIBIL",
        "sibal": "CIBIL", "see bill": "CIBIL", "see built": "CIBIL",
        "hdfc": "HDFC", "icici": "ICICI", "sbi": "SBI",
        "nach": "NACH", "natch": "NACH", "notch": "NACH",
        "upi": "UPI", "gst": "GST", "pan": "PAN", "tan": "TAN",
        "rbi": "RBI", "sebi": "SEBI", "irdai": "IRDAI",
        "nbfc": "NBFC", "npa": "NPA",
        "crore": "crore", "lakh": "lakh",
        "demat": "demat", "d-mat": "demat", "d mat": "demat",
        # More Whisper confusions
        "neft": "NEFT", "rtgs": "RTGS", "imps": "IMPS",
        "cheque": "cheque", "check": "cheque",
        "aadhaar": "Aadhaar", "aadhar": "Aadhaar", "adhar": "Aadhaar",
        "rupee": "rupee", "rupees": "rupees",
        "ip": "IPO", "ipo": "IPO", "i.p.o.": "IPO",
        "sip": "SIP", "s.i.p.": "SIP", "s i p": "SIP",
        "nav": "NAV", "n.a.v.": "NAV",
        "aum": "AUM", "a.u.m.": "AUM",
        "ebitda": "EBITDA", "eps": "EPS",
        "roe": "ROE", "roi": "ROI",
        "mutual fund": "mutual fund", "mutual funds": "mutual funds",
        "fixed deposit": "FD", "fd": "FD", "f.d.": "FD",
        "recurring deposit": "RD", "rd": "RD",
        "provident fund": "PF", "pf": "PF", "epf": "EPF",
        "atm": "ATM", "otp": "OTP", "o.t.p.": "OTP",
        "ifsc": "IFSC", "i.f.s.c.": "IFSC",
    }


# ── Hindi vocabulary extraction ──

def build_hindi_vocab() -> dict:
    """Extract Hindi vocabulary from HuggingFace datasets."""
    vocab = {
        "greetings": [],
        "trivial_phrases": [],
        "currency_words": [],
        "date_words": {"months": [], "relative": [], "days": []},
        "compliance_keywords": {},
        "prohibited_phrases": {},
        "end_call_phrases": [],
        "financial_terms": {"corrections": {}, "acronyms": []},
    }

    # ── skit-ai/skit-s2i: Contains Hindi banking utterances ──
    skit = _load_dataset_safe("skit-ai/skit-s2i", split="train")
    if skit:
        _extract_hindi_from_skit(skit, vocab)

    # ── xlsum Hindi: News summaries with financial vocabulary ──
    xlsum = _load_dataset_safe("csebuetnlp/xlsum", split="train", config="hindi")
    if xlsum:
        _extract_hindi_financial_from_xlsum(xlsum, vocab)

    # Add standard Hindi patterns (universal financial terms, not translations)
    _add_standard_hindi_patterns(vocab)

    return vocab


def _extract_hindi_from_skit(ds, vocab: dict):
    """Extract Hindi phrases from skit-s2i (bilingual Indian banking dataset)."""
    hindi_trivials = Counter()
    hindi_greetings = Counter()

    text_col = None
    for col in ["utterance", "text", "template", "sentence"]:
        if col in ds.column_names:
            text_col = col
            break
    if text_col is None:
        return

    # Hindi script detection: contains Devanagari characters
    hindi_pattern = re.compile(r'[\u0900-\u097F]')

    for row in ds:
        text = row.get(text_col, "").strip()
        if not text or not hindi_pattern.search(text):
            continue

        lower = text.lower().strip()
        words = lower.split()

        if len(words) <= 3:
            hindi_trivials[lower] += 1

        # Check for greeting patterns
        if any(g in lower for g in ("नमस्ते", "नमस्कार", "प्रणाम", "हैलो", "हेलो")):
            hindi_greetings[lower] += 1

    for phrase, count in hindi_trivials.items():
        if count >= 2:
            vocab["trivial_phrases"].append(phrase)
    for phrase, count in hindi_greetings.items():
        if count >= 1:
            vocab["greetings"].append(phrase)

    logger.info(f"skit-s2i Hindi: {len(vocab['trivial_phrases'])} trivials, {len(vocab['greetings'])} greetings")


def _extract_hindi_financial_from_xlsum(ds, vocab: dict):
    """Extract Hindi financial terms from xlsum news summaries."""
    financial_terms = Counter()

    # Hindi financial keywords to search for
    hindi_fin_terms = [
        "रुपये", "रुपया", "करोड़", "लाख", "हजार",
        "ब्याज", "किस्त", "ईएमआई", "ऋण", "कर्ज",
        "बैंक", "खाता", "जमा", "निकासी", "भुगतान",
        "बीमा", "निवेश", "शेयर", "बाजार", "मुनाफा",
    ]

    for row in ds:
        text = row.get("text", "")
        for term in hindi_fin_terms:
            if term in text:
                financial_terms[term] += 1

    vocab["financial_terms"]["acronyms"] = [
        term for term, _ in financial_terms.most_common(50)
    ]
    logger.info(f"xlsum Hindi: extracted {len(vocab['financial_terms']['acronyms'])} financial terms")


def _add_standard_hindi_patterns(vocab: dict):
    """Add standard Hindi financial/compliance patterns."""
    # Greetings (20+ items)
    vocab["greetings"].extend([
        "नमस्ते", "नमस्कार", "प्रणाम", "हैलो", "हेलो",
        "शुभ प्रभात", "शुभ संध्या", "सुप्रभात",
        "शुभ दोपहर", "शुभ रात्रि",
        "नमस्ते जी", "नमस्कार जी", "हैलो जी",
        "हैलो सर", "हैलो मैडम", "नमस्ते सर", "नमस्ते मैडम",
        "शुभ प्रभात सर", "शुभ प्रभात मैडम",
        "सुप्रभात सर", "सुप्रभात मैडम",
        "आदाब", "सत श्री अकाल", "राम राम",
        "जय हिंद", "कैसे हैं आप", "कैसे हो",
    ])

    # Trivial phrases (80+ items)
    vocab["trivial_phrases"].extend([
        "हाँ", "हां", "जी", "जी हाँ", "जी नहीं", "नहीं",
        "ठीक है", "ठीक", "अच्छा", "सही", "बिल्कुल",
        "धन्यवाद", "शुक्रिया", "ओके", "समझ गया", "समझ गयी",
        "हाँ जी", "जी बिल्कुल", "जी सर", "जी मैडम",
        # Extended acknowledgments
        "बहुत अच्छा", "बहुत बढ़िया", "ज़रूर", "बिल्कुल सही",
        "सही बात", "सही है", "पता है", "मालूम है",
        "समझ आ गया", "क्लियर है", "हाँ समझ गया",
        "ठीक है सर", "ठीक है मैडम", "जी ठीक है",
        "अच्छा जी", "हां जी बिल्कुल", "जी हां सर",
        "नहीं सर", "नहीं मैडम", "जी नहीं सर",
        "हां बिल्कुल", "बिल्कुल सर", "ज़रूर सर",
        # Short responses
        "बता दीजिए", "बोलिए", "सुन रहा हूं", "सुन रहे हैं",
        "हां बताइए", "जी बताइए", "हां बोलिए", "जी बोलिए",
        "एक मिनट", "रुकिए", "बस एक मिनट", "ज़रा रुकिए",
        # Confirmations
        "पक्का", "यकीनन", "बेशक", "अवश्य", "ज़रूर करूंगा",
        "कर दूंगा", "कर दूंगी", "भेज दूंगा", "भेज दूंगी",
        "हां कर दूंगा", "जी कर देंगे", "अभी करता हूं",
        # Closing
        "अलविदा", "नमस्ते", "धन्यवाद सर", "धन्यवाद मैडम",
        "शुक्रिया सर", "शुक्रिया मैडम", "बहुत धन्यवाद",
        "बहुत शुक्रिया", "चलिए फिर", "अच्छा चलिए",
        "ठीक है बाय", "अच्छा बाय", "नमस्ते बाय",
        # Polite requests
        "कृपया", "प्लीज", "मेहरबानी", "कृपया बताइए",
        "प्लीज सर", "कृपया सर", "दया करके",
    ])

    # Currency words
    vocab["currency_words"] = [
        {"words": ["रुपये", "रुपया", "रुपए", "रू"], "currency": "INR"},
        {"words": ["लाख"], "currency": "INR_LAKH"},
        {"words": ["करोड़"], "currency": "INR_CRORE"},
        {"words": ["हजार"], "currency": "INR_THOUSAND"},
    ]

    # Date words
    vocab["date_words"] = {
        "months": [
            "जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून",
            "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर",
        ],
        "relative": ["कल", "आज", "परसों", "अगले हफ्ते", "अगले महीने"],
        "days": ["सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार"],
    }

    # Compliance keywords — all 6 categories mirroring English
    vocab["compliance_keywords"] = {
        "collections": {
            "caller_identification": {
                "keywords": ["मेरा नाम", "मैं बोल रहा", "बैंक से", "कंपनी से",
                             "की तरफ से", "मैं कॉल कर रहा", "मैं बोल रही",
                             "हम कॉल कर रहे", "मैं फोन कर रहा"],
                "regulation": "RBI_Fair_Practice_Code_Section_4",
            },
            "purpose_disclosure": {
                "keywords": ["आपके लोन के बारे में", "आपकी किस्त", "बकाया राशि",
                             "भुगतान के संबंध में", "ईएमआई के बारे में",
                             "आपका बकाया", "आपकी पेमेंट", "लोन अकाउंट",
                             "आपके खाते में", "बकाया भुगतान"],
                "regulation": "RBI_Fair_Practice_Code_Section_4",
            },
            "recording_consent": {
                "keywords": ["कॉल रिकॉर्ड", "रिकॉर्ड किया जा रहा", "गुणवत्ता",
                             "यह कॉल रिकॉर्ड", "रिकॉर्डिंग", "क्वालिटी और ट्रेनिंग"],
                "regulation": "IT_Act_2000_Section_43A",
            },
            "mini_miranda": {
                "keywords": ["कर्ज वसूली", "बकाया भुगतान", "ओवरड्यू", "पास्ट ड्यू",
                             "लेट पेमेंट", "डिफॉल्ट", "अतिदेय"],
                "regulation": "RBI_Fair_Practice_Code_Section_8",
            },
        },
        "kyc": {
            "identity_verification": {
                "keywords": ["सत्यापन", "वेरिफिकेशन", "जन्म तिथि", "पैन नंबर",
                             "आधार नंबर", "पुष्टि करें", "कन्फर्म करें",
                             "माता का नाम", "पिता का नाम", "पहचान सत्यापन"],
                "regulation": "RBI_KYC_Master_Direction",
            },
            "data_consent": {
                "keywords": ["सहमति", "अनुमति", "आपकी मंज़ूरी", "अधिकृत",
                             "रज़ामंदी", "परमिशन", "कंसेंट"],
                "regulation": "Digital_Personal_Data_Protection_Act_2023",
            },
            "purpose_of_kyc": {
                "keywords": ["आरबीआई नियम", "नियामक आवश्यकता", "अनिवार्य",
                             "केवाईसी अपडेट", "समय-समय पर", "पुनः सत्यापन"],
                "regulation": "RBI_KYC_Master_Direction_Section_38",
            },
        },
        "consent": {
            "clear_terms": {
                "keywords": ["आप सहमत हैं", "इसका मतलब है", "हाँ कहने से",
                             "अधिकृत कर रहे", "यह अधिकृत करता है"],
                "regulation": "RBI_Digital_Lending_Guidelines",
            },
            "right_to_refuse": {
                "keywords": ["अधिकार है", "बाध्य नहीं", "मना कर सकते",
                             "वैकल्पिक", "आपकी पसंद", "कोई बाध्यता नहीं"],
                "regulation": "Consumer_Protection_Act_2019",
            },
            "cooling_period": {
                "keywords": ["कूलिंग पीरियड", "तीन दिन", "72 घंटे",
                             "सहमति वापस", "रद्द कर सकते", "विचार बदल सकते"],
                "regulation": "RBI_Digital_Lending_Guidelines_Section_6",
            },
        },
        "onboarding": {
            "caller_identification": {
                "keywords": ["मेरा नाम", "बैंक से", "स्वागत है",
                             "चुनने के लिए धन्यवाद"],
                "regulation": "RBI_Fair_Practice_Code_Section_4",
            },
            "product_disclosure": {
                "keywords": ["नियम और शर्तें", "ब्याज दर", "प्रोसेसिंग फीस",
                             "सालाना शुल्क", "चार्जेज", "ईएमआई", "अवधि", "भुगतान"],
                "regulation": "RBI_Fair_Practice_Code_Section_5",
            },
            "grievance_mechanism": {
                "keywords": ["शिकायत", "कस्टमर केयर", "हेल्पलाइन",
                             "लोकपाल", "असंतुष्ट", "समाधान"],
                "regulation": "RBI_Integrated_Ombudsman_Scheme_2021",
            },
        },
        "complaint": {
            "caller_identification": {
                "keywords": ["मेरा नाम", "मैं बोल रहा", "बैंक से"],
                "regulation": "RBI_Fair_Practice_Code_Section_4",
            },
            "complaint_acknowledgment": {
                "keywords": ["शिकायत नंबर", "रेफरेंस नंबर", "टिकट नंबर",
                             "आपकी शिकायत दर्ज", "रजिस्टर किया", "शिकायत आईडी"],
                "regulation": "RBI_Integrated_Ombudsman_Scheme_2021",
            },
            "resolution_timeline": {
                "keywords": ["कार्यदिवस", "दिनों में", "हल करेंगे",
                             "वापस कॉल", "फॉलो अप", "समय सीमा"],
                "regulation": "RBI_Customer_Service_Guidelines",
            },
            "escalation_option": {
                "keywords": ["एस्कलेट", "सुपरवाइज़र", "मैनेजर",
                             "लोकपाल", "बैंकिंग लोकपाल", "उच्च अधिकारी"],
                "regulation": "RBI_Integrated_Ombudsman_Scheme_2021",
            },
        },
        "general": {
            "caller_identification": {
                "keywords": ["मेरा नाम", "मैं बोल रहा", "बैंक से", "कंपनी से"],
                "regulation": "RBI_Fair_Practice_Code_Section_4",
            },
            "recording_consent": {
                "keywords": ["कॉल रिकॉर्ड", "रिकॉर्ड किया जा रहा",
                             "गुणवत्ता और प्रशिक्षण"],
                "regulation": "IT_Act_2000_Section_43A",
            },
        },
    }

    # Prohibited phrases (Hindi — all 6 categories, 40+ total)
    vocab["prohibited_phrases"] = {
        "threats": [
            "पुलिस भेजेंगे", "कानूनी कार्रवाई", "जेल भेजेंगे",
            "घर आएंगे", "सामान जब्त", "नीलामी कर देंगे",
            "रिकवरी एजेंट भेजेंगे", "गिरफ्तारी होगी",
            "वारंट निकलेगा", "कोर्ट से ऑर्डर", "एफआईआर दर्ज",
            "केस दर्ज कर देंगे", "आपकी प्रॉपर्टी कुर्क",
            "बैंक अकाउंट सीज़", "सैलरी से काट लेंगे",
            "ऑफिस में आएंगे", "पड़ोसियों को बता देंगे",
            "रिश्तेदारों को कॉल करेंगे",
        ],
        "coercion": [
            "आपके पास कोई विकल्प नहीं", "अभी भुगतान करो",
            "यह आखिरी मौका", "कॉल करना बंद नहीं करेंगे",
            "आज ही भुगतान करना होगा", "अभी के अभी",
            "कोई बहाना नहीं चलेगा", "और मोहलत नहीं मिलेगी",
            "आज डेडलाइन है", "फाइनल वार्निंग",
            "लास्ट वार्निंग", "पैसे देने ही होंगे",
            "कोई और रास्ता नहीं", "भागोगे कहाँ",
        ],
        "misleading": [
            "सिबिल स्कोर जीरो", "कभी लोन नहीं मिलेगा",
            "नौकरी चली जाएगी", "सैलरी रुक जाएगी",
            "पासपोर्ट कैंसिल", "गिरफ्तार कर लेंगे",
            "आधार ब्लॉक हो जाएगा", "पैन कैंसिल हो जाएगा",
            "सभी बैंकों ने ब्लैकलिस्ट", "आरबीआई ने फ्लैग किया",
            "इंटरपोल को सूचित", "ट्रैवल बैन लगा दिया",
            "बच्चों को लोन नहीं मिलेगा", "परिवार भी जिम्मेदार",
            "पेंशन भी कट जाएगी", "प्रोविडेंट फंड सीज़",
        ],
        "harassment": [
            "शर्म नहीं आती", "बेईमान", "झूठ बोल रहे हो",
            "मेरा समय बर्बाद", "कैसे इंसान हो",
            "शर्म करो", "तमीज़ नहीं है", "बेशर्म",
            "ढोंगी", "फ्रॉड हो तुम", "डिफॉल्टर",
            "आपकी फैमिली को शर्म", "तुम्हारे बच्चे क्या सोचेंगे",
        ],
        "privacy_violation": [
            "तुम्हारे बॉस को बता दिया", "पत्नी से बात की",
            "पड़ोसियों को बताया", "ऑफिस में कॉल किया",
            "रिश्तेदारों को कॉल किया", "ससुराल में फोन किया",
            "तुम्हारी डिटेल्स शेयर की",
        ],
    }

    vocab["end_call_phrases"] = [
        "कॉल बंद करो", "फोन मत करो", "मुझे फोन मत करना",
        "मेरा नंबर हटाओ", "मैं फोन रख रहा हूँ",
        "मुझे परेशान मत करो", "यह कॉल खत्म",
        "दोबारा फोन मत करना", "मुझसे बात मत करो",
        "फोन रख रहा हूँ", "मैं डिस्कनेक्ट कर रहा",
    ]

    # Financial term corrections (Devanagari + common Whisper misrecognitions)
    vocab["financial_terms"]["corrections"] = {
        # Devanagari acronyms
        "ईएमआई": "EMI", "ईएमआइ": "EMI", "ई एम आई": "EMI",
        "केवाईसी": "KYC", "के वाई सी": "KYC", "केवायसी": "KYC",
        "सिबिल": "CIBIL", "सीबिल": "CIBIL", "सी बी आई एल": "CIBIL",
        "एसबीआई": "SBI", "एस बी आई": "SBI",
        "एचडीएफसी": "HDFC", "एच डी एफ सी": "HDFC",
        "आईसीआईसीआई": "ICICI", "आई सी आई सी आई": "ICICI",
        "यूपीआई": "UPI", "यू पी आई": "UPI",
        "जीएसटी": "GST", "जी एस टी": "GST",
        "पैन": "PAN", "पी ए एन": "PAN",
        "आधार": "Aadhaar", "आधार कार्ड": "Aadhaar",
        "आरबीआई": "RBI", "आर बी आई": "RBI",
        "सेबी": "SEBI", "से बी": "SEBI",
        "एनबीएफसी": "NBFC", "एन बी एफ सी": "NBFC",
        "एनपीए": "NPA", "एन पी ए": "NPA",
        "नाच": "NACH", "एन ए सी एच": "NACH",
        "आईएफएससी": "IFSC", "आई एफ एस सी": "IFSC",
        "डीमैट": "demat", "डिमैट": "demat",
        "आईआरडीएआई": "IRDAI",
        # Common Whisper misrecognitions in Hindi speech
        "emi": "EMI", "kyc": "KYC", "otp": "OTP",
        "ओटीपी": "OTP", "ओ टी पी": "OTP",
        "एटीएम": "ATM", "ए टी एम": "ATM",
        "एनईएफटी": "NEFT", "आरटीजीएस": "RTGS",
        "आईएमपीएस": "IMPS",
    }


# ── Tamil vocabulary extraction ──

def build_tamil_vocab() -> dict:
    """Extract Tamil vocabulary from HuggingFace datasets."""
    vocab = {
        "greetings": [],
        "trivial_phrases": [],
        "currency_words": [],
        "date_words": {"months": [], "relative": [], "days": []},
        "compliance_keywords": {},
        "prohibited_phrases": {},
        "end_call_phrases": [],
        "financial_terms": {"corrections": {}, "acronyms": []},
    }

    # ── xlsum Tamil: News summaries ──
    xlsum = _load_dataset_safe("csebuetnlp/xlsum", split="train", config="tamil")
    if xlsum:
        _extract_tamil_financial_from_xlsum(xlsum, vocab)

    # Add standard Tamil patterns
    _add_standard_tamil_patterns(vocab)

    return vocab


def _extract_tamil_financial_from_xlsum(ds, vocab: dict):
    """Extract Tamil financial terms from xlsum news summaries."""
    financial_terms = Counter()

    tamil_fin_terms = [
        "ரூபாய்", "லட்சம்", "கோடி", "ஆயிரம்",
        "வட்டி", "தவணை", "கடன்", "வங்கி", "கணக்கு",
        "காப்பீடு", "முதலீடு", "பங்கு", "சந்தை",
    ]

    for row in ds:
        text = row.get("text", "")
        for term in tamil_fin_terms:
            if term in text:
                financial_terms[term] += 1

    vocab["financial_terms"]["acronyms"] = [
        term for term, _ in financial_terms.most_common(50)
    ]
    logger.info(f"xlsum Tamil: extracted {len(vocab['financial_terms']['acronyms'])} financial terms")


def _add_standard_tamil_patterns(vocab: dict):
    """Add standard Tamil financial/compliance patterns."""
    # Greetings (15+ items)
    vocab["greetings"] = [
        "வணக்கம்", "நல்வரவு",
        "வணக்கம் சார்", "வணக்கம் மேடம்",
        "காலை வணக்கம்", "மாலை வணக்கம்",
        "நல்ல காலை", "நல்ல மாலை",
        "வணக்கம் அண்ணா", "வணக்கம் அக்கா",
        "ஹலோ", "ஹலோ சார்", "ஹலோ மேடம்",
        "காலை வணக்கம் சார்", "மாலை வணக்கம் சார்",
        "வாங்க", "வாங்க சார்", "வருக வருக",
    ]

    # Trivial phrases (40+ items)
    vocab["trivial_phrases"] = [
        "ஆமா", "இல்லை", "சரி", "சரிங்க", "நன்றி",
        "புரிகிறது", "புரிஞ்சுது", "ஓகே", "தெரியும்",
        "ஆமா சார்", "சரி சார்", "நன்றி சார்",
        # Extended acknowledgments
        "சரி சரி", "ஆமா ஆமா", "ம்ம்", "ஆமாம்",
        "கரெக்ட்", "சரியா", "தெரியும் சார்",
        "புரிந்தது", "புரிந்தது சார்", "புரிஞ்சுது சார்",
        "நன்றி மேடம்", "நன்றி அண்ணா", "சரி மேடம்",
        "ஆமா மேடம்", "ஓகே சார்", "ஓகே மேடம்",
        # Confirmations
        "செய்கிறேன்", "செய்யறேன்", "செய்வேன்",
        "பண்றேன்", "பண்ணுறேன்", "சொல்லுங்க",
        "கொஞ்சம் நேரம்", "ஒரு நிமிஷம்",
        "கேட்டுக்கிறேன்", "கேக்குறேன்",
        # Closing
        "போய் வருகிறேன்", "பை", "பை சார்",
        "நன்றி பை", "சரி பை", "வணக்கம் பை",
        "ரொம்ப நன்றி", "மிக்க நன்றி",
        "சரி செய்கிறேன்", "பார்க்கலாம்",
    ]

    # Currency words
    vocab["currency_words"] = [
        {"words": ["ரூபாய்", "ரூ"], "currency": "INR"},
        {"words": ["லட்சம்"], "currency": "INR_LAKH"},
        {"words": ["கோடி"], "currency": "INR_CRORE"},
        {"words": ["ஆயிரம்"], "currency": "INR_THOUSAND"},
    ]

    # Date words
    vocab["date_words"] = {
        "months": [
            "ஜனவரி", "பிப்ரவரி", "மார்ச்", "ஏப்ரல்", "மே", "ஜூன்",
            "ஜூலை", "ஆகஸ்ட்", "செப்டம்பர்", "அக்டோபர்", "நவம்பர்", "டிசம்பர்",
        ],
        "relative": ["நாளை", "இன்று", "நேற்று", "அடுத்த வாரம்", "அடுத்த மாதம்"],
        "days": ["திங்கள்", "செவ்வாய்", "புதன்", "வியாழன்", "வெள்ளி", "சனி", "ஞாயிறு"],
    }

    # Compliance keywords — all 6 categories mirroring English
    vocab["compliance_keywords"] = {
        "collections": {
            "caller_identification": {
                "keywords": ["என் பெயர்", "வங்கியிலிருந்து", "நிறுவனத்திலிருந்து",
                             "தொலைபேசி செய்கிறேன்", "நான் பேசுகிறேன்",
                             "அழைக்கிறேன்"],
                "regulation": "RBI_Fair_Practice_Code_Section_4",
            },
            "purpose_disclosure": {
                "keywords": ["உங்கள் கடன்", "தவணை", "பாக்கி தொகை",
                             "செலுத்தாத தொகை", "இஎம்ஐ", "கடன் கணக்கு",
                             "நிலுவை தொகை"],
                "regulation": "RBI_Fair_Practice_Code_Section_4",
            },
            "recording_consent": {
                "keywords": ["அழைப்பு பதிவு செய்யப்படுகிறது", "பதிவு செய்யப்படும்",
                             "தரம் மற்றும் பயிற்சி"],
                "regulation": "IT_Act_2000_Section_43A",
            },
            "mini_miranda": {
                "keywords": ["கடன் வசூல்", "நிலுவை", "தாமதம்",
                             "காலாவதி", "கடன் தொகை"],
                "regulation": "RBI_Fair_Practice_Code_Section_8",
            },
        },
        "kyc": {
            "identity_verification": {
                "keywords": ["சரிபார்ப்பு", "பிறந்த தேதி", "பான் எண்",
                             "ஆதார் எண்", "உறுதிப்படுத்தவும்"],
                "regulation": "RBI_KYC_Master_Direction",
            },
            "data_consent": {
                "keywords": ["ஒப்புதல்", "அனுமதி", "அங்கீகாரம்"],
                "regulation": "Digital_Personal_Data_Protection_Act_2023",
            },
        },
        "consent": {
            "clear_terms": {
                "keywords": ["நீங்கள் ஒப்புக்கொள்கிறீர்கள்", "இதன் பொருள்",
                             "அங்கீகரிக்கிறீர்கள்"],
                "regulation": "RBI_Digital_Lending_Guidelines",
            },
            "right_to_refuse": {
                "keywords": ["உரிமை", "கட்டாயம் இல்லை", "மறுக்கலாம்",
                             "விருப்பம்", "உங்கள் தேர்வு"],
                "regulation": "Consumer_Protection_Act_2019",
            },
        },
        "onboarding": {
            "product_disclosure": {
                "keywords": ["விதிமுறைகள்", "வட்டி விகிதம்", "கட்டணம்",
                             "இஎம்ஐ", "காலம்", "திருப்பிச் செலுத்துதல்"],
                "regulation": "RBI_Fair_Practice_Code_Section_5",
            },
            "grievance_mechanism": {
                "keywords": ["புகார்", "வாடிக்கையாளர் சேவை", "ஹெல்ப்லைன்",
                             "குறைதீர்ப்பாளர்", "தீர்வு"],
                "regulation": "RBI_Integrated_Ombudsman_Scheme_2021",
            },
        },
        "complaint": {
            "complaint_acknowledgment": {
                "keywords": ["புகார் எண்", "குறிப்பு எண்", "டிக்கெட் எண்",
                             "புகார் பதிவு", "புகார் ஐடி"],
                "regulation": "RBI_Integrated_Ombudsman_Scheme_2021",
            },
            "resolution_timeline": {
                "keywords": ["வேலை நாட்கள்", "தீர்வு", "திரும்ப அழைப்போம்",
                             "பின்தொடர்வு", "கால அவகாசம்"],
                "regulation": "RBI_Customer_Service_Guidelines",
            },
            "escalation_option": {
                "keywords": ["மேலதிகாரி", "மேலாளர்", "குறைதீர்ப்பாளர்",
                             "வங்கி குறைதீர்ப்பாளர்"],
                "regulation": "RBI_Integrated_Ombudsman_Scheme_2021",
            },
        },
        "general": {
            "caller_identification": {
                "keywords": ["என் பெயர்", "வங்கியிலிருந்து"],
                "regulation": "RBI_Fair_Practice_Code_Section_4",
            },
        },
    }

    # Prohibited phrases — all categories (20+ total)
    vocab["prohibited_phrases"] = {
        "threats": [
            "போலீஸ் அனுப்புவோம்", "சட்ட நடவடிக்கை",
            "சிறையில் போவீர்கள்", "வீட்டுக்கு வருவோம்",
            "கைது செய்வோம்", "வாரண்ட் பிறப்பிக்கப்படும்",
            "எஃப்ஐஆர் பதிவு", "சொத்து பறிமுதல்",
            "வங்கி கணக்கு முடக்கப்படும்",
        ],
        "coercion": [
            "வேறு வழி இல்லை", "இப்போதே செலுத்துங்கள்",
            "கடைசி வாய்ப்பு", "இன்றே செலுத்த வேண்டும்",
            "இறுதி எச்சரிக்கை", "தப்பிக்க முடியாது",
            "வேறு வழி கிடையாது",
        ],
        "misleading": [
            "சிபில் ஸ்கோர் பூஜ்ஜியம்", "கடன் கிடைக்காது",
            "வேலை போய்விடும்", "சம்பளம் நிறுத்தப்படும்",
            "பாஸ்போர்ட் ரத்து", "ஆதார் முடக்கப்படும்",
            "பான் ரத்து செய்யப்படும்",
        ],
        "harassment": [
            "வெட்கமில்லையா", "பொய்யர்", "மோசடி",
            "நம்பிக்கையற்றவர்", "ஏமாற்றுகிறீர்கள்",
        ],
        "privacy_violation": [
            "உங்கள் முதலாளிக்கு சொல்வோம்", "அக்கம் பக்கத்தினருக்கு தெரியும்",
            "குடும்பத்தினருக்கு அறிவித்தோம்",
        ],
    }

    vocab["end_call_phrases"] = [
        "தொலைபேசி செய்யாதீர்கள்", "என் நம்பரை நீக்குங்கள்",
        "போன் வைக்கிறேன்", "அழைக்காதீர்கள்",
        "தொந்தரவு செய்யாதீர்கள்", "இந்த அழைப்பு முடிந்தது",
        "மீண்டும் போன் செய்யாதீர்கள்",
    ]

    # Financial term corrections (Tamil + common Whisper misrecognitions)
    vocab["financial_terms"]["corrections"] = {
        "இஎம்ஐ": "EMI", "இ எம் ஐ": "EMI",
        "கேஒய்சி": "KYC", "கே ஒய் சி": "KYC",
        "சிபில்": "CIBIL", "சி பி ஐ எல்": "CIBIL",
        "யுபிஐ": "UPI", "யு பி ஐ": "UPI",
        "ஜிஎஸ்டி": "GST", "ஜி எஸ் டி": "GST",
        "ஆர்பிஐ": "RBI", "ஆர் பி ஐ": "RBI",
        "பான்": "PAN", "பி ஏ என்": "PAN",
        "ஆதார்": "Aadhaar", "ஆதார் கார்டு": "Aadhaar",
        "என்பிஎஃப்சி": "NBFC", "என்பிஏ": "NPA",
        "ஐஎஃப்எஸ்சி": "IFSC", "நாச்": "NACH",
        "டிமேட்": "demat", "ஏடிஎம்": "ATM",
        "ஓடிபி": "OTP", "ஓ டி பி": "OTP",
        "என்இஎஃப்டி": "NEFT", "ஆர்டிஜிஎஸ்": "RTGS",
    }


# ── Fraud / Scam Indicator Vocabulary ──

def build_fraud_indicators() -> dict:
    """Extract scam/fraud indicator phrases from HuggingFace datasets.

    Downloads real labeled scam conversations and extracts vocabulary
    categorized by fraud type and tactic.

    Datasets used:
    - BothBosu/Scammer-Conversation (1K, 10 scam types)
    - BothBosu/multi-agent-scam-conversation (1.6K, 8 scam types)
    - shakeleoatmeal/phone-scam-detection-synthetic (1.8K, 3 types, 6 subtlety levels)
    """
    indicators = {
        "remote_access_tools": [],
        "urgency_pressure": [],
        "credential_requests": [],
        "authority_impersonation": [],
        "vague_identity": [],
        "financial_threats": [],
        "social_engineering": [],
        "scam_type_keywords": {},
        "source_datasets": [],
    }

    # Dataset 1: BothBosu/Scammer-Conversation (broadest — 10 scam types)
    ds1 = _load_dataset_safe("BothBosu/Scammer-Conversation", split="train")
    if ds1:
        _extract_scam_phrases_bothbosu(ds1, indicators, "BothBosu/Scammer-Conversation")

    # Dataset 2: BothBosu/multi-agent-scam-conversation
    ds2 = _load_dataset_safe("BothBosu/multi-agent-scam-conversation", split="train")
    if ds2:
        _extract_scam_phrases_multiagent(ds2, indicators, "BothBosu/multi-agent-scam-conversation")

    # Dataset 3: shakeleoatmeal/phone-scam-detection-synthetic
    ds3 = _load_dataset_safe("shakeleoatmeal/phone-scam-detection-synthetic", split="train")
    if ds3:
        _extract_scam_phrases_synthetic(ds3, indicators, "shakeleoatmeal/phone-scam-detection-synthetic")

    # Add baseline single-keyword/short-phrase indicators
    _ensure_baseline_indicators(indicators)

    # Deduplicate all lists
    for key in indicators:
        if isinstance(indicators[key], list) and indicators[key] and isinstance(indicators[key][0], str):
            indicators[key] = sorted(set(indicators[key]))

    logger.info(
        f"Fraud indicators: "
        f"{len(indicators['remote_access_tools'])} remote_access, "
        f"{len(indicators['urgency_pressure'])} urgency, "
        f"{len(indicators['credential_requests'])} credential, "
        f"{len(indicators['authority_impersonation'])} authority, "
        f"{len(indicators['vague_identity'])} vague_id, "
        f"{len(indicators['financial_threats'])} threats, "
        f"{len(indicators['social_engineering'])} social_eng, "
        f"{len(indicators['scam_type_keywords'])} scam types"
    )

    return indicators


def _extract_scam_phrases_bothbosu(ds, indicators: dict, ds_name: str):
    """Extract scam indicator phrases from BothBosu/Scammer-Conversation.

    Each row has: conversation (string), label (0=normal, 1=scam).
    Uses TF difference: phrases frequent in scam but rare in normal.
    """
    scam_phrases = Counter()
    normal_phrases = Counter()

    text_col = "conversation" if "conversation" in ds.column_names else ds.column_names[0]
    label_col = "label" if "label" in ds.column_names else None

    scam_count = 0
    normal_count = 0

    for row in ds:
        text = row.get(text_col, "")
        if not isinstance(text, str) or not text:
            continue
        label = row.get(label_col, 1) if label_col else 1
        is_scam = label == 1

        # Extract phrases (3-grams and 4-grams)
        words = text.lower().split()
        for n in (3, 4, 5):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i+n])
                if is_scam:
                    scam_phrases[phrase] += 1
                else:
                    normal_phrases[phrase] += 1

        if is_scam:
            scam_count += 1
        else:
            normal_count += 1

    # Find phrases significantly more common in scam than normal
    _categorize_overrepresented_phrases(scam_phrases, normal_phrases, scam_count, normal_count, indicators)
    indicators["source_datasets"].append(ds_name)
    logger.info(f"{ds_name}: analyzed {scam_count} scam + {normal_count} normal conversations")


def _extract_scam_phrases_multiagent(ds, indicators: dict, ds_name: str):
    """Extract scam indicator phrases from multi-agent scam conversations."""
    scam_phrases = Counter()
    normal_phrases = Counter()

    text_col = "dialogue" if "dialogue" in ds.column_names else ds.column_names[0]
    label_col = "labels" if "labels" in ds.column_names else "label"

    scam_count = 0
    for row in ds:
        text = row.get(text_col, "")
        if not isinstance(text, str) or not text:
            continue
        label = row.get(label_col, 1)
        is_scam = label == 1

        words = text.lower().split()
        for n in (3, 4, 5):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i+n])
                if is_scam:
                    scam_phrases[phrase] += 1
                else:
                    normal_phrases[phrase] += 1

        if is_scam:
            scam_count += 1

    _categorize_overrepresented_phrases(scam_phrases, normal_phrases, scam_count, max(len(ds) - scam_count, 1), indicators)
    indicators["source_datasets"].append(ds_name)
    logger.info(f"{ds_name}: analyzed {scam_count} scam conversations")


def _extract_scam_phrases_synthetic(ds, indicators: dict, ds_name: str):
    """Extract scam phrases from phone-scam-detection-synthetic dataset.

    Has subtlety levels — extract phrases weighted by subtlety
    (direct scams have the most obvious vocabulary).
    """
    scam_phrases = Counter()
    normal_phrases = Counter()

    text_col = "dialogue" if "dialogue" in ds.column_names else ds.column_names[0]
    label_col = "label" if "label" in ds.column_names else None

    scam_count = 0
    for row in ds:
        text = row.get(text_col, "")
        if not isinstance(text, str) or not text:
            continue
        label = row.get(label_col, 1) if label_col else 1
        is_scam = label == 1

        words = text.lower().split()
        for n in (3, 4, 5):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i+n])
                if is_scam:
                    scam_phrases[phrase] += 1
                else:
                    normal_phrases[phrase] += 1

        if is_scam:
            scam_count += 1

    _categorize_overrepresented_phrases(scam_phrases, normal_phrases, scam_count, max(len(ds) - scam_count, 1), indicators)
    indicators["source_datasets"].append(ds_name)
    logger.info(f"{ds_name}: analyzed {scam_count} scam conversations")


def _categorize_overrepresented_phrases(
    scam_phrases: Counter,
    normal_phrases: Counter,
    scam_count: int,
    normal_count: int,
    indicators: dict,
):
    """Find phrases overrepresented in scam vs normal, categorize by fraud type."""

    # Keyword patterns for categorization
    REMOTE_ACCESS_KW = {"anydesk", "teamviewer", "remote", "screen", "share", "desktop",
                        "ammyy", "ultraviewer", "rustdesk", "download", "install", "app"}
    URGENCY_KW = {"immediately", "urgent", "right now", "hurry", "quickly", "asap",
                  "expire", "limited", "last chance", "act fast", "running out",
                  "don't wait", "time-sensitive", "within the hour", "deadline"}
    CREDENTIAL_KW = {"password", "pin", "otp", "verification code", "social security",
                     "ssn", "account number", "routing number", "card number",
                     "cvv", "bank details", "login", "credentials", "access code"}
    AUTHORITY_KW = {"irs", "ssa", "social security administration", "federal",
                    "government", "department of", "law enforcement", "police",
                    "fbi", "treasury", "tax department", "internal revenue",
                    "immigration", "customs", "rbi", "sebi"}
    VAGUE_ID_KW = {"online company", "finance department", "technical department",
                   "security department", "accounts department", "support team",
                   "head office", "main office", "central office"}
    THREAT_KW = {"arrest", "warrant", "jail", "prison", "legal action", "lawsuit",
                 "penalty", "fine", "suspend", "cancel", "freeze", "block",
                 "terminated", "revoked", "prosecute", "criminal"}
    SOCIAL_ENG_KW = {"trust me", "help you", "protect", "secure", "safety",
                     "compromised", "hacked", "breach", "unauthorized", "suspicious",
                     "fraudulent", "verify your identity", "confirm your",
                     "for your protection", "security check", "routine check"}

    # Minimum frequency in scam corpus
    min_freq = max(3, scam_count // 100)

    for phrase, count in scam_phrases.most_common(5000):
        if count < min_freq:
            continue

        # Compute overrepresentation ratio
        normal_freq = normal_phrases.get(phrase, 0)
        scam_rate = count / max(scam_count, 1)
        normal_rate = normal_freq / max(normal_count, 1)
        ratio = scam_rate / max(normal_rate, 0.0001)

        if ratio < 2.0:
            continue  # Not significantly overrepresented in scam

        phrase_words = set(phrase.split())

        # Categorize based on keyword overlap
        if phrase_words & REMOTE_ACCESS_KW:
            indicators["remote_access_tools"].append(phrase)
        elif phrase_words & URGENCY_KW:
            indicators["urgency_pressure"].append(phrase)
        elif phrase_words & CREDENTIAL_KW:
            indicators["credential_requests"].append(phrase)
        elif phrase_words & AUTHORITY_KW:
            indicators["authority_impersonation"].append(phrase)
        elif phrase_words & VAGUE_ID_KW:
            indicators["vague_identity"].append(phrase)
        elif phrase_words & THREAT_KW:
            indicators["financial_threats"].append(phrase)
        elif phrase_words & SOCIAL_ENG_KW:
            indicators["social_engineering"].append(phrase)


def _ensure_baseline_indicators(indicators: dict):
    """Add baseline single-keyword and short-phrase fraud indicators.

    The n-gram extraction from datasets only produces 3-5 word phrases
    but misses critical single-keyword indicators like tool names.
    This ensures essential baseline indicators are always present.
    """
    # Remote access tools — single keywords that datasets miss
    baseline_remote = [
        "anydesk", "teamviewer", "ammyy", "ultraviewer", "rustdesk",
        "share screen", "screen share", "remote access", "remote desktop",
        "download this app", "install this app", "quick support",
        "logmein", "connectwise", "splashtop", "chrome remote desktop",
        "let me control", "give me access", "share your screen",
        "i will fix it remotely", "remote support session",
    ]
    # Urgency pressure — short phrases
    baseline_urgency = [
        "act now", "don't delay", "time is running out", "expires today",
        "last chance", "final notice", "immediate action required",
        "within the next hour", "before midnight", "offer expires",
        "limited time", "act immediately", "don't wait", "right away",
        "this is urgent", "emergency", "critical situation",
        "you must act now", "window is closing",
    ]
    # Credential requests — specific asks
    baseline_creds = [
        "your password", "your pin", "your otp", "verification code",
        "card number", "cvv number", "bank details", "account number",
        "routing number", "social security number", "aadhaar number",
        "pan number", "login credentials", "your username",
        "mother maiden name", "security question", "access code",
        "one time password", "enter your pin", "share your otp",
        "tell me your password", "what is your pin",
    ]
    # Authority impersonation
    baseline_authority = [
        "from the irs", "social security administration", "federal agent",
        "law enforcement", "department of treasury", "tax department",
        "internal revenue service", "immigration officer", "customs officer",
        "rbi officer", "sebi officer", "income tax department",
        "cyber crime cell", "cbi officer", "enforcement directorate",
        "narcotics bureau", "government official", "from the bank",
    ]
    # Vague identity — currently empty, fill it
    baseline_vague_id = [
        "online company", "finance department", "technical department",
        "security department", "accounts department", "support team",
        "head office", "main office", "central office",
        "technical support", "customer service center", "processing unit",
        "verification department", "compliance department", "fraud department",
        "recovery department", "legal department", "collections department",
        "we are from", "authorized agency", "government authorized",
        "bank authorized", "empaneled agency",
    ]
    # Financial threats
    baseline_threats = [
        "arrest warrant", "you will be arrested", "legal case",
        "criminal charges", "jail time", "prison sentence",
        "freeze your account", "suspend your account", "cancel your card",
        "block your account", "seize your property", "garnish your wages",
        "penalty charges", "heavy fine", "blacklist you",
        "default notice", "legal notice served", "court summons",
    ]
    # Social engineering
    baseline_social_eng = [
        "trust me", "for your safety", "for your protection",
        "your account is compromised", "suspicious activity",
        "unauthorized transaction", "security breach", "hacked",
        "verify your identity", "confirm your details",
        "routine security check", "mandatory verification",
        "you have been selected", "you won a prize",
        "refund is pending", "cashback offer", "insurance claim",
        "tax refund pending", "unclaimed funds",
    ]
    # Scam type keywords — was empty
    baseline_scam_types = {
        "tech_support_scam": ["virus detected", "computer infected", "microsoft calling",
                              "windows support", "fix your computer", "malware found"],
        "irs_tax_scam": ["unpaid taxes", "tax evasion", "irs warrant",
                         "back taxes", "tax fraud", "tax lien"],
        "bank_fraud_scam": ["suspicious transaction", "account compromised",
                            "unauthorized withdrawal", "card cloned", "phishing attempt"],
        "loan_scam": ["pre-approved loan", "guaranteed approval", "no credit check",
                      "upfront fee", "processing fee required", "advance payment"],
        "investment_scam": ["guaranteed returns", "double your money", "risk free",
                            "insider tip", "once in a lifetime", "exclusive opportunity"],
        "identity_theft": ["update your kyc", "aadhaar linking mandatory",
                           "pan verification required", "verify or account blocked"],
    }

    for phrase in baseline_remote:
        if phrase not in indicators["remote_access_tools"]:
            indicators["remote_access_tools"].append(phrase)
    for phrase in baseline_urgency:
        if phrase not in indicators["urgency_pressure"]:
            indicators["urgency_pressure"].append(phrase)
    for phrase in baseline_creds:
        if phrase not in indicators["credential_requests"]:
            indicators["credential_requests"].append(phrase)
    for phrase in baseline_authority:
        if phrase not in indicators["authority_impersonation"]:
            indicators["authority_impersonation"].append(phrase)
    for phrase in baseline_vague_id:
        if phrase not in indicators["vague_identity"]:
            indicators["vague_identity"].append(phrase)
    for phrase in baseline_threats:
        if phrase not in indicators["financial_threats"]:
            indicators["financial_threats"].append(phrase)
    for phrase in baseline_social_eng:
        if phrase not in indicators["social_engineering"]:
            indicators["social_engineering"].append(phrase)

    # Merge scam type keywords
    if not indicators.get("scam_type_keywords"):
        indicators["scam_type_keywords"] = {}
    for scam_type, keywords in baseline_scam_types.items():
        existing = indicators["scam_type_keywords"].get(scam_type, [])
        for kw in keywords:
            if kw not in existing:
                existing.append(kw)
        indicators["scam_type_keywords"][scam_type] = existing


# ── Main ──

def main():
    """Build vocabulary files for all languages."""
    output_dir = Path("data/vocab")
    output_dir.mkdir(parents=True, exist_ok=True)

    # English
    logger.info("=" * 60)
    logger.info("Building English vocabulary...")
    en_vocab = build_english_vocab()
    en_path = output_dir / "en.json"
    with open(en_path, "w", encoding="utf-8") as f:
        json.dump(en_vocab, f, indent=2, ensure_ascii=False)
    logger.info(f"English vocab saved to {en_path}")
    _log_vocab_stats("English", en_vocab)

    # Hindi
    logger.info("=" * 60)
    logger.info("Building Hindi vocabulary...")
    hi_vocab = build_hindi_vocab()
    hi_path = output_dir / "hi.json"
    with open(hi_path, "w", encoding="utf-8") as f:
        json.dump(hi_vocab, f, indent=2, ensure_ascii=False)
    logger.info(f"Hindi vocab saved to {hi_path}")
    _log_vocab_stats("Hindi", hi_vocab)

    # Tamil
    logger.info("=" * 60)
    logger.info("Building Tamil vocabulary...")
    ta_vocab = build_tamil_vocab()
    ta_path = output_dir / "ta.json"
    with open(ta_path, "w", encoding="utf-8") as f:
        json.dump(ta_vocab, f, indent=2, ensure_ascii=False)
    logger.info(f"Tamil vocab saved to {ta_path}")
    _log_vocab_stats("Tamil", ta_vocab)

    # Fraud indicators (English, from scam conversation datasets)
    logger.info("=" * 60)
    logger.info("Building fraud indicator vocabulary...")
    fraud_vocab = build_fraud_indicators()
    fraud_path = output_dir / "fraud_indicators.json"
    with open(fraud_path, "w", encoding="utf-8") as f:
        json.dump(fraud_vocab, f, indent=2, ensure_ascii=False)
    logger.info(f"Fraud indicators saved to {fraud_path}")

    logger.info("=" * 60)
    logger.info("Vocabulary build complete!")


def _log_vocab_stats(lang: str, vocab: dict):
    """Log statistics about extracted vocabulary."""
    g = len(vocab.get("greetings", []))
    t = len(vocab.get("trivial_phrases", []))
    c = len(vocab.get("currency_words", []))
    p = sum(len(v) for v in vocab.get("prohibited_phrases", {}).values())
    ck = sum(len(v) for v in vocab.get("compliance_keywords", {}).values())
    ft = len(vocab.get("financial_terms", {}).get("acronyms", []))
    fc = len(vocab.get("financial_terms", {}).get("corrections", {}))
    logger.info(
        f"  {lang}: {g} greetings, {t} trivials, {c} currency sets, "
        f"{p} prohibited, {ck} compliance categories, {ft} fin terms, {fc} corrections"
    )


if __name__ == "__main__":
    main()
