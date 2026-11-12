"""audit_clean.py - VIMA3YA ticket-triage training-data audit and cleaning.

Usage:  python audit_clean.py --train data/train.jsonl --dev data/dev.jsonl --out_dir .

Writes: train_clean.jsonl, train_raw.jsonl (raw baseline for experiment 1), cleaning_log.csv
Uses only train.jsonl and SCHEMA.md rules. dev.jsonl is read ONLY to find and drop dev tickets
that leaked into train (never for labels).
"""
import re, datetime as dt

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
MON_RE = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"


def split_ticket(text):
    head, _, body = text.partition("\n\n")
    m = re.search(r"Received: (\d{4}-\d{2}-\d{2})", head)
    recv = dt.date.fromisoformat(m.group(1)) if m else None
    src = re.search(r"Source: (\S+)", head)
    own = "\n".join(l for l in body.split("\n") if not l.lstrip().startswith(">"))
    return recv, (src.group(1) if src else None), body, own


# ---------- txn_id ----------
TXN_RE = re.compile(r"(?i)(?<![a-z0-9])n\s*p[\s\-_.#:]*((?:\d[\s\-]*){10})(?![\d])")


def txn_ids(own):
    out = []
    for m in TXN_RE.finditer(own):
        d = re.sub(r"\D", "", m.group(1))
        if len(d) == 10:
            out.append("NP" + d)
    return out


# ---------- amount ----------
NUM = r"(?<![\d.,])(\d[\d,]*(?:\.\d+)?)(?![\d]|[.,]\d)"
UNIT = r"\s*(k|lakhs?|lacs?)?\b"
AMT_PATTERNS = [
    re.compile(r"(?i)(?:rs\.?|inr|₹)\s*" + NUM + UNIT),
    re.compile(r"(?i)" + NUM + r"\s*(k|lakhs?|lacs?)\b"),
    re.compile(r"(?i)" + NUM + r"()\s*(?:rupees|rupaye|rs\b)"),
    re.compile(r"(?i)(?:amount:?|total|rakam|amount (?:is|was)|\bit was for|\bit is)\s+" + NUM + r"()(?!\s*(?:k|lakhs?|lacs?)\b)(?!\.\d)\b"),
    re.compile(r"(?i)(?<![#])" + NUM + r"()(?!\s*(?:k|lakhs?|lacs?)\b)\s+(?:kat gaye|is the amount|was the value|in total|ka transaction|ka tha|tha\b)"),
]
EXCLUDE_BEFORE = re.compile(r"(?i)(balance is|limit is|abhi|fee|charge of)\s*$")
EXCLUDE_AFTER = re.compile(r"(?i)^\s*(rupees\s+|rupaye\s+)?(right now|pade hain)")


def to_paise(num, unit):
    v = float(num.replace(",", ""))
    unit = (unit or "").lower()
    if unit == "k":
        v *= 1000
    elif unit.startswith(("lakh", "lac")):
        v *= 100000
    return int(round(v * 100))


def amounts(own):
    found = {}
    for pat in AMT_PATTERNS:
        for m in pat.finditer(own):
            s, e = m.span()
            before, after = own[max(0, s - 22):s], own[e:e + 22]
            excluded = bool(EXCLUDE_BEFORE.search(before) or EXCLUDE_AFTER.search(after))
            # skip numbers that are part of app versions / ticket numbers / dates
            if re.search(r"(?i)(version\s*|#|ticket number\s*|n\s*p[\s\-]*|ref(?: no)?\s*)$", before):
                continue
            key = (s // 1, )
            p = to_paise(m.group(1), m.group(2))
            found.setdefault(p, excluded)
            found[p] = found[p] and excluded  # kept if any mention is non-excluded
    good = sorted({p for p, ex in found.items() if not ex})
    bad = sorted({p for p, ex in found.items() if ex})
    return good, bad


# ---------- date ----------
def most_recent(day, month, recv):
    for y in (recv.year, recv.year - 1):
        try:
            d = dt.date(y, month, day)
        except ValueError:
            continue
        if d <= recv:
            return d
    return None


def dates(own, recv):
    out = []
    low = own.lower()
    for m in re.finditer(r"(?<![\d.])(\d{1,2})[/-](\d{1,2})[/-](\d{4}|\d{2})(?!\d)(?!\.\d)", own):
        y = int(m.group(3)); y = y + 2000 if y < 100 else y
        try:
            out.append(dt.date(y, int(m.group(2)), int(m.group(1))))
        except ValueError:
            out.append(("bad", m.group(0)))
    for m in re.finditer(r"(?i)(?<![\d])(\d{1,2})(?:st|nd|rd|th)?\s+" + MON_RE + r"\b(?:,?\s*(\d{4}))?", own):
        d, mo, y = int(m.group(1)), MONTHS[m.group(2).lower()[:3]], m.group(3)
        out.append(dt.date(int(y), mo, d) if y else most_recent(d, mo, recv))
    for m in re.finditer(r"(?i)\b" + MON_RE + r"\s+(\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s*(\d{4}))?", own):
        mo, d, y = MONTHS[m.group(1).lower()[:3]], int(m.group(2)), m.group(3)
        out.append(dt.date(int(y), mo, d) if y else most_recent(d, mo, recv))
    if re.search(r"\bday before yesterday\b|\bparso\b", low):
        out.append(recv - dt.timedelta(2))
    elif re.search(r"\byesterday\b|\bkal\b", low):
        out.append(recv - dt.timedelta(1))
    if re.search(r"\btoday\b|\baaj\b", low):
        out.append(recv)
    for m in re.finditer(r"(?i)(\d+)\s+(?:days ago|din pehle)", own):
        out.append(recv - dt.timedelta(int(m.group(1))))
    return out


# ---------- channel ----------
def channels(own):
    low = own.lower()
    c = set()
    if re.search(r"\bupi\b", low): c.add("upi")
    if re.search(r"net\s*banking|internet banking", low): c.add("netbanking")
    if re.search(r"\bwallet\b", low): c.add("wallet")
    if re.search(r"\b(credit|debit)\s+card\b|card payment|\bvisa\b|mastercard|(?<!scratch )(?<!srcatch )\bcard\b", low): c.add("card")
    return c


# ---------- language ----------
HI = set("""hai hain tha thi hua hui gaya gayi gaye kar karo karna kiya kiye raha rahi rahe nahi nahin
mera mere meri mujhe maine ka ke ki ko se mein bhi abhi kab kyun kya kitna kinta wapas paise rupaye chahiye
baar bhai ji jaldi diya liya pehle pichla rakam yeh woh aaj kal parso din bol jaunga dunga
chuka kat atka atak kaise karun dhanyavaad""".split())


def hinglish_score(own):
    words = re.findall(r"[a-z]+", own.lower())
    return sum(w in HI for w in words)


# ---------- flags ----------
REPEAT = re.compile(r"(?i)third time|already written \d+ times|contacted support about this before|again and again|"
                    r"teesri baar|pehle bhi complaint|baar baar|second time|second complaint|multiple times about this|"
                    r"already (?:complained|raised|reported)")
LEGAL = re.compile(r"(?i)\brbi\b|ombudsman|consumer court|lawyer|legal notice|legal action")
FRAUD = re.compile(r"(?i)did not make|didn'?t make|not recogni[sz]e|don'?t recogni[sz]e|wasn'?t me|was not me|"
                   r"not authori[sz]ed|unauthori[sz]ed|maine nahi kiya|maine koi payment nahi|nahi kiya|"
                   r"someone (?:stole|used|took|has)|stolen|hack|fraud|scam|without my (?:permission|knowledge)|"
                   r"never made|i didn'?t do|not mine")


def expected_priority(cat, amount_paise, legal):
    if cat == "fraud" or (amount_paise is not None and amount_paise >= 5_000_000) or legal:
        return "P1"
    if (cat in ("payment_failed", "refund") and amount_paise is not None and amount_paise >= 500_000) \
            or cat == "account_access":
        return "P2"
    return "P3"


# =====================================================================================
# Cleaning
# =====================================================================================
import json, csv, argparse, collections

KEYS = ["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]
CATS = {"payment_failed", "refund", "fraud", "account_access", "kyc", "offers", "other"}
TXN_CATS = {"payment_failed", "refund", "fraud", "offers"}
OLD_PRIO = {"high": "P1", "medium": "P2", "low": "P3"}  # legacy v1 key/scale found in 70 rows


def norm_text(t):
    return re.sub(r"\s+", " ", t).strip()


def is_junk(own):
    return len(own.strip()) < 20 and bool(re.fullmatch(r"(?i)[\s]*(test|asdf|test test 123|asdf asdf)?[\s]*", own))


def validate(lab):
    """Strict schema check of a cleaned label. Returns list of problems."""
    p = []
    if list(lab) != KEYS: p.append("keys")
    if lab.get("category") not in CATS: p.append("category")
    if lab.get("priority") not in ("P1", "P2", "P3"): p.append("priority")
    a = lab.get("amount_paise")
    if not (a is None or type(a) is int): p.append("amount type")
    d = lab.get("txn_date")
    if d is not None:
        try: dt.date.fromisoformat(d)
        except Exception: p.append("date")
    if lab.get("txn_id") is not None and not re.fullmatch(r"NP\d{10}", lab["txn_id"]): p.append("txn_id")
    if lab.get("channel") not in (None, "upi", "card", "netbanking", "wallet"): p.append("channel")
    if lab.get("language") not in ("en", "hi-en"): p.append("language")
    if type(lab.get("needs_human")) is not bool: p.append("needs_human")
    if lab.get("category") in ("account_access", "kyc", "other") and any(lab[k] is not None for k in ("amount_paise", "txn_date", "txn_id", "channel")):
        p.append("txn fields on non-txn category")
    return p


def clean(train_path, dev_path, out_dir):
    dev_texts = {norm_text(json.loads(l)["messages"][1]["content"]) for l in open(dev_path, encoding="utf-8")}
    log, kept, raw = [], [], []
    seen = {}
    for line in open(train_path, encoding="utf-8"):
        r = json.loads(line); rid = r["id"]; text = r["messages"][1]["content"]
        recv, src, body, own = split_ticket(text)
        content = r["messages"][2]["content"]
        try:
            lab = json.loads(content)
        except json.JSONDecodeError:
            log.append((rid, "drop", "assistant label is truncated / invalid JSON")); continue
        in_dev = norm_text(text) in dev_texts
        if not in_dev:
            raw.append(r)                       # raw baseline: everything trainable, minus dev leakage
        if in_dev:
            log.append((rid, "drop", "ticket text identical to a dev.jsonl ticket (dev leakage)")); continue
        if is_junk(own):
            log.append((rid, "drop", f"junk ticket body {own.strip()!r} with arbitrary label")); continue

        reasons = []
        if content != content.strip():
            reasons.append("label had leading/trailing whitespace")
        if "prio" in lab:
            old = lab.pop("prio"); lab["priority"] = OLD_PRIO[old]
            reasons.append(f"renamed legacy key prio={old!r} -> priority={lab['priority']!r}")
        if isinstance(lab.get("txn_date"), str) and re.fullmatch(r"\d{2}/\d{2}/\d{4}", lab["txn_date"]):
            dd, mm, yy = lab["txn_date"].split("/"); new = f"{yy}-{mm}-{dd}"
            reasons.append(f"txn_date {lab['txn_date']} (DD/MM/YYYY) -> {new}"); lab["txn_date"] = new
        if isinstance(lab.get("txn_id"), str) and not re.fullmatch(r"NP\d{10}", lab["txn_id"]):
            new = "NP" + re.sub(r"\D", "", lab["txn_id"])
            reasons.append(f"txn_id {lab['txn_id']!r} -> {new} (normalize)"); lab["txn_id"] = new
        good, _ = amounts(own)
        a = lab.get("amount_paise")
        if isinstance(a, int) and len(good) == 1 and good[0] != a and abs(good[0] - a * 100) < 100:
            reasons.append(f"amount_paise {a} is rupees not paise; text amount -> {good[0]}"); lab["amount_paise"] = good[0]
        legal = bool(LEGAL.search(own))
        exp_p = expected_priority(lab["category"], lab["amount_paise"], legal)
        if lab["priority"] != exp_p:
            why = ("fraud" if lab["category"] == "fraud" else "legal threat" if legal else
                   "amount>=Rs50,000" if (lab["amount_paise"] or 0) >= 5_000_000 else
                   "account_access" if lab["category"] == "account_access" else
                   "payment_failed/refund amount>=Rs5,000" if exp_p == "P2" else "no P1/P2 condition holds")
            reasons.append(f"priority {lab['priority']} -> {exp_p} per SCHEMA ({why}); needs_human={lab['needs_human']} agrees")
            lab["priority"] = exp_p
        lab = {k: lab[k] for k in KEYS}
        probs = validate(lab)
        assert not probs, (rid, probs, lab)
        key = norm_text(text)
        if key in seen:
            first_id, first_lab = seen[key]
            assert first_lab == lab, ("conflicting duplicate", rid, first_id)
            log.append((rid, "drop", f"duplicate of {first_id} (same ticket text and label)")); continue
        seen[key] = (rid, lab)
        if reasons:
            log.append((rid, "fix", "; ".join(reasons)))
        r = {"id": rid, "messages": [r["messages"][0], r["messages"][1],
                                     {"role": "assistant", "content": json.dumps(lab, ensure_ascii=False)}]}
        kept.append(r)

    import os
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/train_clean.jsonl", "w", encoding="utf-8") as f:
        for r in kept: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{out_dir}/train_raw.jsonl", "w", encoding="utf-8") as f:
        for r in raw: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{out_dir}/cleaning_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "action", "reason"]); w.writerows(log)
    return kept, raw, log


def verify(kept):
    """Re-derive every rule-checkable field on the cleaned data; all must agree."""
    bad = collections.Counter()
    for r in kept:
        lab = json.loads(r["messages"][2]["content"]); recv, _, _, own = split_ticket(r["messages"][1]["content"])
        cat = lab["category"]
        ids = txn_ids(own); good, _ = amounts(own)
        ds = sorted({d for d in dates(own, recv) if isinstance(d, dt.date)}); ch = channels(own)
        if cat in TXN_CATS:
            if lab["txn_id"] != (ids[0] if ids else None): bad["txn_id"] += 1
            if len(good) <= 1 and lab["amount_paise"] != (good[0] if good else None): bad["amount_paise"] += 1
            if len(ds) <= 1 and lab["txn_date"] != (ds[0].isoformat() if ds else None): bad["txn_date"] += 1
            if len(ch) <= 1 and lab["channel"] != (next(iter(ch)) if ch else None): bad["channel"] += 1
        if lab["language"] != ("hi-en" if hinglish_score(own) >= 2 else "en"): bad["language"] += 1
        if lab["priority"] != expected_priority(cat, lab["amount_paise"], bool(LEGAL.search(own))): bad["priority"] += 1
        if lab["needs_human"] != (lab["priority"] == "P1" or bool(REPEAT.search(own))): bad["needs_human"] += 1
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.jsonl"); ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out_dir", default=".")
    a = ap.parse_args()
    kept, raw, log = clean(a.train, a.dev, a.out_dir)
    c = collections.Counter(x[1] for x in log)
    print(f"train rows in: {sum(1 for _ in open(a.train))} | kept (clean): {len(kept)} | raw baseline rows: {len(raw)}")
    print(f"log entries: {dict(c)}")
    groups = collections.Counter()
    for _, act, why in log:
        for part in why.split("; "):
            key = re.sub(r"\d", "#", part.split(" (")[0].split(" ->")[0].split("=")[0].split(";")[0])[:60]
            groups[(act, key)] += 1
    for (act, k), n in sorted(groups.items(), key=lambda x: -x[1]): print(f"  {act:5} {n:4}  {k}")
    print("verification disagreements on cleaned data:", dict(verify(kept)) or "none")
