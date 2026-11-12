"""augment.py - E4: threshold-focused counterfactual amount augmentation.

Takes cleaned training rows (payment_failed / refund), rewrites ONLY the transaction amount to a value near
a SCHEMA.md priority threshold (Rs 5,000 or Rs 50,000), keeps the ticket's own amount format, and recomputes
amount_paise, priority and needs_human with the SCHEMA rules from audit_clean.py. No other model is used.

Usage: python augment.py --clean clean/train_clean.jsonl --out clean/train_aug.jsonl --n 400 --seed 42
"""
import json, re, random, argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_clean as A

KEYS = A.KEYS


def fmt_indian(n):
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:]); head = head[:-2]
    if head: parts.insert(0, head)
    return ",".join(parts) + "," + tail


def amount_spans(own):
    """Return [(start, end, paise, num_text, unit)] for non-excluded amount mentions (number+unit part only)."""
    out = []
    for pat in A.AMT_PATTERNS:
        for m in pat.finditer(own):
            s, e = m.span()
            before, after = own[max(0, s - 22):s], own[e:e + 22]
            if A.EXCLUDE_BEFORE.search(before) or A.EXCLUDE_AFTER.search(after):
                continue
            if re.search(r"(?i)(version\s*|#|ticket number\s*|n\s*p[\s\-]*|ref(?: no)?\s*)$", before):
                continue
            unit = (m.group(2) or "")
            end = m.end(2) if unit else m.end(1)
            out.append((m.start(1), end, A.to_paise(m.group(1), unit), m.group(1), unit))
    return out


def render_amount(rupees_x100, num_text, unit):
    """Format a new amount (in paise) in the same style as the original mention."""
    if unit.lower() == "k":
        r = rupees_x100 // 100
        r = round(r / 100) * 100                       # k-style amounts are multiples of 100 rupees
        return f"{r/1000:.1f}".rstrip("0").rstrip(".") + "k", r * 100
    if unit:                                           # lakh style: never near thresholds -> skip
        return None, None
    has_dec = "." in num_text
    has_comma = "," in num_text
    rupees, paise = divmod(rupees_x100, 100)
    if not has_dec:
        paise = 0
    txt = fmt_indian(rupees) if has_comma else str(rupees)
    if has_dec:
        txt += f".{paise:02d}"
    return txt, rupees * 100 + paise


def sample_amount(rng):
    """Paise value near a threshold. Includes exact edge values (4,999 / 5,000 / 49,999 / 50,000)."""
    r = rng.random()
    if r < 0.12:
        return rng.choice([499900, 500000, 4999900, 5000000, 500100, 5000100])
    if r < 0.56:
        return rng.randint(350000, 650000) + rng.choice([0, 0, 25, 50, 75, 99])
    return rng.randint(4000000, 6000000) + rng.choice([0, 0, 25, 50, 75, 99])


def augment(rows, n, seed=42):
    rng = random.Random(seed)
    cands = []
    for r in rows:
        text = r["messages"][1]["content"]
        lab = json.loads(r["messages"][2]["content"])
        if lab["category"] not in ("payment_failed", "refund") or lab["amount_paise"] is None:
            continue
        if "\n>" in text:                              # keep it simple: no quoted threads
            continue
        recv, src, body, own = A.split_ticket(text)
        spans = [s for s in amount_spans(own) if s[2] == lab["amount_paise"]]
        if len({(s[0], s[1]) for s in spans}) != 1:   # exactly one mention of the labelled amount
            continue
        cands.append((r, lab, spans[0]))
    rng.shuffle(cands)
    out, i = [], 0
    while len(out) < n and i < 10 * len(cands):
        r, lab, (s, e, paise, num_text, unit) = cands[i % len(cands)]; i += 1
        new_txt, new_paise = render_amount(sample_amount(rng), num_text, unit)
        if new_txt is None or new_paise == lab["amount_paise"]:
            continue
        text = r["messages"][1]["content"]
        head, sep, body = text.partition("\n\n")
        # 'own' == body when there are no quoted lines, so spans index into body directly
        new_body = body[:s] + new_txt + body[e:]
        new_text = head + sep + new_body
        recv, _, _, own = A.split_ticket(new_text)
        good, _ = A.amounts(own)
        if good != [new_paise]:                        # extractor must read back exactly the new amount
            continue
        new = dict(lab)
        new["amount_paise"] = new_paise
        new["priority"] = A.expected_priority(new["category"], new_paise, bool(A.LEGAL.search(own)))
        new["needs_human"] = new["priority"] == "P1" or bool(A.REPEAT.search(own))
        new = {k: new[k] for k in KEYS}
        assert not A.validate(new), (r["id"], A.validate(new))
        out.append({"id": f"{r['id']}-aug{len(out):03d}",
                    "messages": [r["messages"][0], {"role": "user", "content": new_text},
                                 {"role": "assistant", "content": json.dumps(new, ensure_ascii=False)}]})
    return out, len(cands)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="clean/train_clean.jsonl"); ap.add_argument("--out", default="clean/train_aug.jsonl")
    ap.add_argument("--n", type=int, default=400); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.clean, encoding="utf-8")]
    aug, n_cands = augment(rows, a.n, a.seed)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows + aug: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import collections
    pri = collections.Counter(json.loads(r["messages"][2]["content"])["priority"] for r in aug)
    print(f"eligible source tickets: {n_cands} | augmented rows: {len(aug)} | priority mix: {dict(pri)}")
    print(f"written: {len(rows)} clean + {len(aug)} augmented = {len(rows) + len(aug)} rows -> {a.out}")
    print("verification disagreements on augmented rows:", dict(A.verify(aug)) or "none")
