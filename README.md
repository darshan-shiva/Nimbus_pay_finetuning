# NimbusPay ticket triage — LoRA fine-tune of Qwen2.5-1.5B-Instruct

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Loaded in 4-bit?** **No.** Trained and run with the base model in **fp16** (LoRA adapter weights fp32). 4-bit QLoRA was evaluated as experiment E2 and not chosen.
- **Best result (dev, 200 tickets):** **93.5% exact match, 98.7% mean field accuracy, 100% valid JSON** (final notebook run on a Colab T4; E4 configuration: cleaned data + 400 threshold-augmented rows, LoRA r=16).

## Repository

| Path | What it is |
|---|---|
| `adapter/` | Final LoRA adapter in PEFT format (`adapter_config.json`, `adapter_model.safetensors`) |
| `notebook.ipynb` | End-to-end notebook with outputs: cleaning → baselines → training → dev eval → error analysis → test predictions → experiment table |
| `predictions_test.jsonl` | One line per test ticket: `{"id": ..., "output": "<raw model output>"}` |
| `cleaning_log.csv` | Every dropped or fixed training row with the reason (`id, action, reason`) |
| `audit_clean.py` | Data audit + cleaning (rules from SCHEMA.md; self-verifying) |
| `augment.py` | E4 threshold-focused augmentation (rule-based, no other model) |
| `report.md` | Findings, experiments, error analysis, next steps, AI-tool disclosure |
| `logs/` | Experiment logs (`experiments.jsonl` = Colab E0–E3, `experiments_kaggle.jsonl` = Kaggle E4) and `final_result.json` |

Training data and base-model weights are not included.

## Reproducing the predictions

```python
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float16, device_map={"": 0})
model = PeftModel.from_pretrained(base, "adapter").eval()          # not merged

row = json.loads(open("data/test_inputs.jsonl").readline())
text = tok.apply_chat_template(row["messages"][:2], tokenize=False, add_generation_prompt=True)
enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**enc, max_new_tokens=200, do_sample=False, temperature=None, top_p=None, top_k=None,
                     repetition_penalty=1.0, pad_token_id=tok.pad_token_id)
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())
```

- Input: exactly the `system` and `user` messages, Qwen's own chat template (it has a system role).
- **Greedy decoding.** Qwen's `generation_config.json` enables sampling and a repetition penalty by default, so `do_sample=False` and `repetition_penalty=1.0` are passed explicitly.
- `predictions_test.jsonl` was generated one ticket at a time (batch size 1, no padding) with exactly this call. Only `.strip()` is applied to the output.
- Library versions: transformers 5.22.2, peft 0.21.2, trl 1.17.1, accelerate 1.15.0, bitsandbytes 0.50.2, datasets 5.2.0 on a Colab T4; the exact torch build is printed in the notebook's first cell and saved in `logs/final_result.json`.
- Reproducibility check (notebook section 10): batched and batch-1 outputs identical for 400/400 test rows; adapter reloaded from disk regenerated 25/25 random rows exactly.

## Running the notebook

Fresh Colab T4 runtime, about 1h30m (test predictions at batch size 1 take ~35 min). Put `training_assessment.zip`, `audit_clean.py`, `augment.py`, `experiments.jsonl` and `experiments_kaggle.jsonl` in `MyDrive/vima3ya/`, then Runtime → Run all. Outputs are written to `MyDrive/vima3ya/final/`. The notebook also runs on a Kaggle T4 (inputs as a dataset).
