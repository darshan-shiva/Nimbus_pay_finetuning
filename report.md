# Report: NimbusPay ticket triage with a LoRA-tuned Qwen2.5-1.5B

**Result:** `Qwen/Qwen2.5-1.5B-Instruct` (fp16 base, LoRA r=16): **93.5% exact match, 98.7% mean field accuracy, 100% valid JSON on dev**, up from 0% for the untuned model.

## 1. Data audit (train.jsonl: 1,445 → 1,296 clean rows)

**Method:**
1. Structural checks: the label parses as JSON, has exactly the 8 keys, valid values and types, and null transaction fields for non-transaction categories.
2. An independent rule-based labeller written from SCHEMA.md re-derives txn_id, amount, date, channel, language, priority and needs_human from the ticket text, and was compared against every label.
3. Every disagreement was read before deciding. Often the rule was wrong ("scratch card" is not `channel: card`; "second complaint" counts as repeat contact), so I fixed the rule, not the label.
4. Category was checked by grouping identical intent sentences plus a keyword pass, reading every flagged ticket.

| Finding | Rows | Action | Evidence |
|---|---|---|---|
| Assistant label is truncated JSON | 25 | drop | `json.loads` fails; rebuilding them would use my rules' output, not real labels |
| Ticket identical to a dev ticket (30 exact, 7 whitespace-only) | 37 | drop | training on dev would inflate every dev score |
| Duplicate tickets (same label) | 77 | drop | normalised-text duplicates |
| Junk tickets ("test", "asdf", empty) | 10 | drop | arbitrary labels; none in dev/test |
| Legacy key `prio` with high/medium/low | 70 | fix → `priority` P1/P2/P3 | mapping matches the spec in 70/70 rows |
| Wrong priority | 60 | fix to spec | base rates (account_access is P2 in 168 rows vs P3 in 6); `needs_human` agrees with the spec priority in all 60 |
| `amount_paise` stored in rupees | 25 | fix ×100 | e.g. "2.2 lakh" stored as 220000; priority already matched the true amount |
| `txn_date` as DD/MM/YYYY | 30 | fix → ISO | values right, format wrong |
| txn_id not normalised (`NP-…`, `np 12345 67890`) | 20 | fix | digits always right |

- Fields with no label errors: category, language, channel, needs_human, date values.
- After cleaning, re-deriving every rule-checkable field gives **zero disagreements**.
- Every decision is in `cleaning_log.csv`; dev and test were never modified.
- Borderline case kept: two "can't log in as myself" tickets labelled `kyc`.

## 2. Baselines (dev, untuned model, required inference procedure)

| Baseline | Exact | Field acc | Valid JSON |
|---|---|---|---|
| 1: fixed system prompt | 0.0% | 0.0% | 0% |
| 2: + SCHEMA.md in the prompt (reference only) | 0.0% | 0.0% | 0% |

- Baseline 1 invents its own keys and wraps the output in a ```json fence.
- Baseline 2 knows the rules, but also fences every answer, which the strict scorer counts as invalid.
- With fences stripped (diagnostic only), baseline 2 reaches 61.6% field accuracy and 0.5% exact. Its weakest fields are needs_human (30%) and priority (35%), the fields that combine several rules.

## 3. Training setup

- **Model:** Qwen2.5-1.5B-Instruct. It has a system role and fits a T4 in fp16.
- **Loss masking:** TRL `SFTTrainer` with prompt/completion data, so loss is only on the JSON answer plus `<|im_end|>`. Verified by decoding the labelled tokens.
- **LoRA:** all 7 linear projections, r=16, alpha=32, dropout 0.05 (18.5M trainable parameters; r=64 would exceed the 50M cap).
- **Precision:** fp16 base with fp32 adapter weights (the T4 has no bf16).
- **Optimisation:** lr 2e-4 cosine, 5% warmup, effective batch 16, 3 epochs, gradient checkpointing, seed 42.
- **`max_length=2304`:** answers are at most 83 tokens, but emails with quoted threads reach 2,215 tokens, and TRL's default of 1024 truncates from the right, which would cut off the answer.

## 4. Experiments (dev; one change at a time against E0)

| Experiment | What changed | Exact | Field acc | Priority | Peak GPU GiB | Train min |
|---|---|---|---|---|---|---|
| E0 reference | clean data, fp16 LoRA, r=16 | 92.0 | 98.5 | 95.0 | 9.47* | 23.3 |
| E1 | raw data instead of clean | 85.0 | 97.1 | 87.5 | 5.90 | 25.0 |
| E2 | 4-bit QLoRA (NF4) | 90.0 | 98.2 | 92.5 | 5.54 | 28.3 |
| E3a | rank 8 (9.2M params) | 88.5 | 98.0 | 93.0 | 5.76 | 22.6 |
| E3b | rank 32 (36.9M params) | 93.5 | 98.6 | 95.5 | 6.23 | 22.8 |
| E4 | + 400 threshold-augmented rows | 93.0 | 98.8 | 96.0 | 5.90 | 26.5† |
| **Final** | E4 configuration, notebook run | **93.5** | **98.7** | **95.5** | **5.90** | **25.7** |

\* E0's peak includes about 3.5 GiB left on the GPU by a crashed validation cell. E1 (same configuration, raw data) is the clean fp16-LoRA reference.
† E4 ran on a Kaggle T4 (same GPU, different torch build) after Colab's free GPU quota ran out. E0–E3 and the final run are on Colab T4.

- **E1, data quality.** Raw data costs 7 points of exact match, concentrated in priority (−7.5), the field with the most corrupted labels. Fields with no label errors don't move. Training loss stays about 4× higher, because contradictory labels can't be fitted.
- **E2, QLoRA.** 4-bit saved only 0.36 GiB (6%), took 21% longer, and scored slightly lower. For a 1.5B model the frozen weights are small (about 3 GB in fp16), the k-bit preparation upcasts the large embedding to fp32, and peak memory is dominated by activations on ~2.2k-token rows. QLoRA pays off for 7B+ models, not here.
- **E3, rank.** r=8 is clearly weaker. r=16 and r=32 differ by about 3 tickets, which is within noise.
- **E4, augmentation.** E0's priority errors were all near the ₹5k/₹50k thresholds, so I rewrote only the amount of 400 clean tickets to near-threshold values (same format) and recomputed priority and needs_human with the spec's rules. Field accuracy rose slightly (98.5 → 98.7–98.9) and priority by at most 1 point. E4 also trained 31% more steps, so **the targeted hypothesis is not confirmed**.
- **Noise.** One seed per configuration, 200 dev tickets: one ticket is 0.5 points and the standard error of exact match is about ±2 points. The same E4 configuration ran three times with 93.5, 93.0 and 93.5. Only E1 and E3a differ from E0 clearly beyond noise.
- **Final choice.** E4's configuration over E3b, based on field accuracy and a 74 MB adapter (r=32 would be 148 MB), not on a significant difference. Choosing on dev makes the dev score slightly optimistic.

## 5. Error analysis (final model, 200 dev tickets, 13 failures)

- **No JSON errors:** all 200 outputs are valid.
- **Weakest fields:** priority 95.5%, category 98.0%, needs_human 98.0%, txn_date 99.0%. Channel and language are 100%.
- **Weakest categories (exact match):** fraud 80.0%, payment_failed 90.6%, refund 90.9%. All other categories are 100%.
- **By language:** Hinglish tickets reach 89.8% exact vs 94.7% for English, although Hinglish *detection* is perfect (0 language errors).
- **Errors by type:** priority 9, category 4, needs_human 4, date 2, amount 1, txn_id 1, Hinglish detection 0.

| Ticket | Expected → predicted | Why it failed |
|---|---|---|
| dv-00127 | priority P3 → P2 | ₹4,836 is under the ₹5k threshold; amount read correctly, comparison wrong |
| dv-00088 | priority P3 → P2 | Hinglish refund, "Rakam 3685 rupaye": same <₹5k comparison error |
| dv-00180 | priority P3 → P2 | ₹3,130 read correctly despite a ₹2,41,670 balance distractor, but over-prioritised |
| dv-00056 | priority P2 → P1, needs_human F → T | ₹44,206.75 is under ₹50k; treated as ≥₹50k |
| dv-00037 | priority P1 → P2, needs_human T → F | "284669 rupees" (no commas) is ≥₹50k; amount right, threshold missed |
| dv-00093 | fraud → refund (P1 → P3) | "A random charge got added to my card": unrecognised charge read as a billing issue |
| dv-00157 | fraud → account_access (P1 → P2) | "Kisi ne OTP maang ke… maine nahi kiya": the word OTP pulled it to account_access |
| dv-00010 | payment_failed → fraud (P3 → P1) | "I didn't see the money I added to my card": missing top-up read as theft |
| dv-00177 | payment_failed → fraud | "a red flag on my top up": surface word "flag" suggests fraud |
| dv-00006 | date 2026-06-30 → 2026-07-01 | "day before yesterday" resolved as yesterday |
| dv-00090 | date 2026-07-10 → 2026-07-11 | Hinglish "parso" (day before yesterday) resolved as yesterday |
| dv-00147 | txn_id NP1954629979 → NP19546299979 | spaced ID `np 19546 29979` copied with an extra 9 (11 digits) |
| dv-00124 | amount null → 702676793.6 (and P3 → P2) | no amount in the ticket; the model turned the txn_id digits into a float amount and raised the priority |

**Patterns.** Priority errors are threshold comparisons on correctly extracted amounts. Six of nine involve amounts in the ₹3–5k or ₹40–50k bands, and the ₹3–5k cases are always over-prioritised to P2. Category errors are all fraud versus a neighbouring category, decided by surface words ("OTP", "flag", "random charge"). Relative dates and spaced IDs fail on copying or counting, not understanding.

## 6. What I would do next

1. Several seeds per configuration, to separate real effects from the ±2-point noise.
2. A step-matched control for E4 (clean data trained for the same number of steps) to isolate the augmentation effect.
3. Denser augmentation in the ₹3–5k band, fraud-vs-payment_failed minimal pairs, "parso"/"day before yesterday" variants and spaced IDs. All of these are generated by rules from train.jsonl, as in E4.
4. The test set is "deliberately harder": build a harder dev slice (long quoted threads, noisy formatting) to measure that directly.
5. Try Qwen2.5-3B with the same recipe; it fits a T4 with LoRA.

## 7. Engineering notes

- Colab ships torchao 0.10, which PEFT rejects; the notebook uninstalls it.
- A crashed validation cell left about 3.5 GiB on the GPU and inflated E0's measured peak memory (diagnosed with `torch.cuda.memory_allocated`). Runs now free the trainer explicitly.
- QLoRA first failed because adapter weights became bf16 under the fp16 grad scaler. Trainable parameters are now re-cast to fp32 after the trainer is built.
- `predictions_test.jsonl` is generated at batch size 1. Batched and batch-1 outputs were identical for 400/400 rows. Reloading the adapter from disk and regenerating 25 random rows reproduced the file exactly.

## 8. AI tools used

- **Claude (Anthropic):** helped plan the stages; wrote the audit/cleaning script (`audit_clean.py`), the augmentation script (`augment.py`) and the Colab/Kaggle notebooks; debugged the torchao, memory-leak and bf16 issues; drafted this report and the README from my measured results.
- **My own work:** ran every experiment on Colab and Kaggle, reviewed the cleaning decisions and the cleaning log, chose the experiments and the final model, and checked all numbers against the notebook outputs.
