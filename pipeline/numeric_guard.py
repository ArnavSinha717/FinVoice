"""Keep arithmetic and currency out of the LLM's hands.

The summary model was asked to "cite specific dollar amounts" on rupee-denominated
calls and duly invented conversions — one run rendered 3,40,000 rupees as
"approximately $52,808 USD", off by more than tenfold. Fabricated figures in a
compliance record are a serious failure mode, and they were intermittent, which is
worse than consistently wrong.

The fix is structural rather than prompt-tuning: the LLM writes prose, and every
number in that prose must trace back to something the extraction layer actually
found in the audio. Anything that does not is removed. Derived quantities (totals,
sums of promised payments) are computed here, in code, from the extracted entities.
"""

import re
from dataclasses import dataclass

# Currency mentions that are never valid on an INR call unless the extraction layer
# actually found that currency. Conversions are the specific failure observed.
# Two shapes are needed. The model emits "$52,808 USD" on one run and
# "approximately 51,679 USD" — no symbol at all — on the next, and a
# symbol-only pattern silently misses the second.
_CURRENCY_WORD = r"(?:USD|EUR|GBP|JPY|AUD|CAD|dollars?|euros?|pounds?|yen)"
_APPROX = r"(?:approximately|approx\.?|about|roughly|around|nearly|~|=|equivalent to)"
_MAGNITUDE = r"(?:million|billion|thousand|k|m|bn)"

_FOREIGN_CURRENCY = re.compile(
    rf"\(?\s*{_APPROX}?\s*"
    rf"(?:"
    rf"(?:US\s*)?[$€£¥]\s*[\d,]+(?:\.\d+)?\s*{_MAGNITUDE}?\s*{_CURRENCY_WORD}?"
    rf"|"
    rf"[\d,]+(?:\.\d+)?\s*{_MAGNITUDE}?\s*{_CURRENCY_WORD}"
    rf")"
    rf"\s*\)?",
    re.IGNORECASE,
)

# Any number that looks like a monetary quantity in the summary text.
_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")


@dataclass
class GuardResult:
    text: str
    removed: list[str]
    unsupported: list[str]

    @property
    def changed(self) -> bool:
        return bool(self.removed or self.unsupported)


def _canonical(value: str) -> str:
    """Normalise a numeric string for comparison: '3,40,000' and '340000' match."""
    return value.replace(",", "").replace(" ", "").lstrip("0") or "0"


def allowed_numbers(entities: list, segments: list) -> set[str]:
    """Every number the pipeline can actually vouch for.

    Sourced from extracted financial entities and from the transcript itself — if a
    figure was spoken, quoting it is fine; if it was not, the model invented it.
    """
    allowed: set[str] = set()

    for e in entities or []:
        for field in ("value", "raw_text"):
            raw = e.get(field) if isinstance(e, dict) else getattr(e, field, None)
            if raw is None:
                continue
            for m in _NUMBER.finditer(str(raw)):
                allowed.add(_canonical(m.group()))

    for seg in segments or []:
        text = seg.get("text", "") if isinstance(seg, dict) else ""
        for m in _NUMBER.finditer(text):
            allowed.add(_canonical(m.group()))

    # Small integers are ordinary prose ("three points", "2 speakers"), not claims.
    for n in range(0, 101):
        allowed.add(str(n))
    return allowed


def scrub(text: str, entities: list, segments: list) -> GuardResult:
    """Remove invented currency conversions and report unsupported figures."""
    if not text:
        return GuardResult(text or "", [], [])

    removed: list[str] = []
    allowed = allowed_numbers(entities, segments)

    # Foreign-currency amounts are only kept if that exact figure was extracted.
    def _drop_foreign(match: re.Match) -> str:
        span = match.group().strip()
        digits = _NUMBER.search(span)
        if digits and _canonical(digits.group()) in allowed:
            return match.group()
        removed.append(span)
        return ""

    cleaned = _FOREIGN_CURRENCY.sub(_drop_foreign, text)

    # Tidy what the removal left behind: empty parentheses, floating punctuation,
    # and dangling conjunctions from phrases like "X rupees or about 4,100 dollars".
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\b(?:or|and|roughly|approximately|about|around|i\.e\.|equal to)\s*"
                     r"(?=[.,;:]|$)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[,\-–—]\s*(?=[.;:])", "", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

    # Any remaining figure that is not traceable to an extracted entity or to the
    # transcript is the model doing arithmetic. Observed: a model computed a
    # "remaining balance" of 249,200 where 340,000 - 10,000 = 330,000. Such figures
    # are neutralised rather than left to read as fact; the correct derived values
    # are computed in compute_totals() and carried on the record separately.
    unsupported: list[str] = []

    def _neutralise(match: re.Match) -> str:
        token = match.group()
        if _canonical(token) in allowed:
            return token
        unsupported.append(token)
        return "[unverified]"

    cleaned = _NUMBER.sub(_neutralise, cleaned)
    cleaned = re.sub(r"[₹$€£¥]\s*\[unverified\]", "[unverified]", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return GuardResult(cleaned, removed, unsupported)


def compute_totals(entities: list) -> dict:
    """Derived figures, computed in code rather than asked of the model."""
    def _val(e, field):
        return e.get(field) if isinstance(e, dict) else getattr(e, field, None)

    # Only sum things where a sum is meaningful. Totalling reference numbers or
    # account numbers produces a large, confident, meaningless figure.
    summable = {
        "payment_amount", "emi_amount", "loan_amount", "penalty_amount",
        "outstanding_amount", "currency_amount", "late_fee",
    }
    amounts: dict[str, list[float]] = {}
    for e in entities or []:
        etype = str(_val(e, "entity_type") or "")
        if etype not in summable:
            continue
        raw = str(_val(e, "value") or "")
        m = re.search(r"\d[\d,]*(?:\.\d+)?", raw)
        if not m:
            continue
        try:
            amounts.setdefault(etype, []).append(float(m.group().replace(",", "")))
        except ValueError:
            continue

    totals = {f"{k}_total": round(sum(v), 2) for k, v in amounts.items() if v}
    totals.update({f"{k}_count": len(v) for k, v in amounts.items() if v})
    if "payment_amount" in amounts:
        totals["payment_amount_max"] = round(max(amounts["payment_amount"]), 2)
    return totals


def scrub_all(summary: str, outcomes: list[str], actions: list[str],
              entities: list, segments: list) -> tuple[str, list[str], list[str], dict]:
    """Apply the guard to every free-text field the model produced."""
    s = scrub(summary, entities, segments)
    outs, acts, removed, unsupported = [], [], list(s.removed), list(s.unsupported)
    for item in outcomes or []:
        r = scrub(item, entities, segments)
        outs.append(r.text); removed += r.removed; unsupported += r.unsupported
    for item in actions or []:
        r = scrub(item, entities, segments)
        acts.append(r.text); removed += r.removed; unsupported += r.unsupported

    audit = {
        "removed_currency_claims": removed,
        "unsupported_figures": sorted(set(unsupported)),
    }
    return s.text, outs, acts, audit


# ── Prompt-echo guard ─────────────────────────────────────────────────────────
# The summary prompt uses bracketed placeholders ("[Key financial metric ...]") to
# show the model the output shape. Small models sometimes echo those placeholders
# back instead of filling them in, and the result rendered as the headline Summary
# on the call detail page. Strip anything that is still a placeholder, and rebuild
# a factual summary from extracted data when nothing usable survives.

_PLACEHOLDER = re.compile(r"\[[^\]]{8,}\]")


# Headings the prompt asks the model to cover. When the model has nothing to say it
# sometimes emits the headings alone — "Amounts and Balances: Guidance or Forecasts:
# Commitments Made:" — which is long enough to pass a naive length check.
# The heading may carry trailing description before its colon — the model emits
# "Key financial metric or result with exact numbers:" — so allow words in between.
_PROMPT_HEADINGS = re.compile(
    r"\b(?:amounts and balances|guidance or forecasts?|commitments made|"
    r"regulatory matters|outlook|key financial metric|important strategic decision|"
    r"strategic decision|risk factor|regulatory development|"
    r"forward.looking statement|guidance or forward.looking|next actions|"
    r"key outcomes|summary|outcomes|actions)"
    r"[^:\n]{0,70}:",
    re.IGNORECASE,
)


def strip_placeholders(text: str) -> str:
    """Remove un-filled prompt scaffolding from model output.

    Returns "" when nothing of substance survives, so the caller can fall back to a
    summary built from extracted data rather than rendering scaffolding to a user.
    """
    if not text:
        return ""
    cleaned = _PLACEHOLDER.sub("", text)
    cleaned = _PROMPT_HEADINGS.sub("", cleaned)
    cleaned = re.sub(r"\*\*", "", cleaned)
    cleaned = re.sub(r"\s*:\s*(?=:|$)", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -–—:*")

    # Substance test: enough words, and actual prose rather than a list of labels.
    words = [w for w in re.split(r"\W+", cleaned) if w]
    if len(words) < 12:
        return ""
    return cleaned


def fallback_summary(record: dict) -> str:
    """A factual summary assembled in code, for when the model returns nothing usable."""
    bits = []
    ctype = record.get("call_type", "general")
    dur = record.get("duration_seconds") or 0
    bits.append(f"A {ctype} call of {int(dur // 60)}m {int(dur % 60)}s.")

    amounts = [e for e in record.get("financial_entities", [])
               if "amount" in str(e.get("entity_type", ""))]
    if amounts:
        shown = ", ".join(str(e.get("raw_text", "")).strip() for e in amounts[:4] if e.get("raw_text"))
        if shown:
            bits.append(f"Amounts discussed: {shown}.")

    viol = [c for c in record.get("compliance_checks", []) if not c.get("passed")]
    if viol:
        bits.append(f"{len(viol)} compliance violation(s): "
                    f"{', '.join(c.get('check_name', '?') for c in viol[:4])}.")
    else:
        bits.append("No compliance violations detected.")

    if record.get("pii_count"):
        bits.append(f"{record['pii_count']} PII entities detected and masked.")
    risk = record.get("overall_risk_level")
    if risk:
        bits.append(f"Overall risk assessed as {risk}.")
    return " ".join(bits)
