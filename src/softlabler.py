
#ONLY WORKS IN KAGGLE NOTEBOOK WITH QWEN2.5 7b-instruct ADDED

!pip install -q -U bitsandbytes>=0.46.1

import os
import re
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_CSV       = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv")
CHECKPOINT_PATH = Path("/kaggle/working/soft_labels_checkpoint.parquet")
OUTPUT_PATH     = Path("/kaggle/working/soft_labels_final.parquet")
MODEL_ID        = "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1"

# ── 12 conditions ─────────────────────────────────────────────────────────────
CONDITIONS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Bakers", "Contusion", "Fracture",   # ← Bakers no apostrophe
]

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(TRAIN_CSV)
df = df.rename(columns={"Baker's": "Bakers"})         # ← rename in dataframe
df[CONDITIONS] = df[CONDITIONS].fillna(0.5)

to_label        = df[["StudyInstanceUID", "Report"]].reset_index(drop=True)
existing_labels = df[["StudyInstanceUID"] + CONDITIONS].copy()

print(f"Total reports : {len(to_label)}")
print(f"Empty reports : {to_label['Report'].isna().sum()}")
print(f"\nLabel distribution:")
print(existing_labels[CONDITIONS].apply(pd.Series.value_counts).fillna(0).astype(int))

# ── 4-bit quant ───────────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ── GPU check ─────────────────────────────────────────────────────────────────
assert torch.cuda.is_available(), "No GPU — switch runtime!"
print(f"\nGPU : {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Load model ────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.float16,
)
model.eval()
print(f"Model loaded ✓  VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a radiology report parser. Your job is to read knee MRI reports 
and extract findings for exactly 12 conditions. The report may be in any language.

For each condition output exactly one value:
1   = clearly present / diagnosed
0   = clearly absent / explicitly denied
0.5 = not mentioned or ambiguous

Return ONLY a JSON object with exactly these 12 keys, no explanation:
{"ACL": ..., "MCL": ..., "Medial Meniscus": ..., "Lateral Meniscus": ...,
 "Medial OA": ..., "Lateral OA": ..., "PF OA": ..., "Effusion": ...,
 "Synovitis": ..., "Bakers": ..., "Contusion": ..., "Fracture": ...}"""

def build_prompt(report: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Report:\n{report.strip()}"},
    ]

def parse_response(text: str) -> dict | None:
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        raw = json.loads(match.group())
        result = {}
        for c in CONDITIONS:
            v = float(raw.get(c, 0.5))
            result[c] = v if v in (0.0, 0.5, 1.0) else 0.5
        return result
    except Exception:
        return None

MAX_REPORT_TOKENS = 768

@torch.inference_mode()
def run_qwen(report: str) -> dict:
    messages = build_prompt(report)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_REPORT_TOKENS + 200,   # ← truncation added here
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,                           # ← suppresses warning
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response   = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return parse_response(response)

# ── Checkpoint helpers ────────────────────────────────────────────────────────
def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        done_df = pd.read_parquet(CHECKPOINT_PATH)
        done    = done_df.set_index("StudyInstanceUID")[CONDITIONS].to_dict(orient="index")
        print(f"Resuming — {len(done)} reports already done.")
        return done
    print("Starting fresh.")
    return {}

def save_checkpoint(results: dict) -> None:
    df_ck = pd.DataFrame.from_dict(results, orient="index")
    df_ck.index.name = "StudyInstanceUID"
    df_ck.reset_index().to_parquet(CHECKPOINT_PATH, index=False)

SAVE_EVERY = 50

# ── Inference loop ────────────────────────────────────────────────────────────
results      = load_checkpoint()
already_done = set(results.keys())
pending      = to_label[~to_label["StudyInstanceUID"].isin(already_done)].reset_index(drop=True)
print(f"Pending: {len(pending)}  |  Done: {len(already_done)}")

failed = []

for i, row in tqdm(pending.iterrows(), total=len(pending), desc="Labeling"):
    sid     = row["StudyInstanceUID"]
    report  = row["Report"]
    t0      = time.time()
    labels  = run_qwen(report)
    elapsed = time.time() - t0

    if labels is None:
        labels = run_qwen(report)
    if labels is None:
        labels = {c: 0.5 for c in CONDITIONS}
        failed.append(sid)

    results[sid] = labels

    if (i + 1) % SAVE_EVERY == 0:
        save_checkpoint(results)
        print(f"  ✓ checkpoint saved at {i+1} | last report: {elapsed:.1f}s")

save_checkpoint(results)
print(f"\nDone. Total: {len(results)}  |  Failed: {len(failed)}")

# ── Merge Qwen labels with golden labels ─────────────────────────────────────
qwen_df      = pd.DataFrame.from_dict(results, orient="index")
qwen_df.index.name = "StudyInstanceUID"
qwen_df      = qwen_df.reset_index()
qwen_indexed = qwen_df.set_index("StudyInstanceUID")

final_df = existing_labels.copy()

for c in CONDITIONS:
    mask  = existing_labels[c] == 0.5
    sids  = existing_labels.loc[mask, "StudyInstanceUID"]
    valid = sids[sids.isin(qwen_indexed.index)]
    final_df.loc[valid.index, c] = qwen_indexed.loc[valid.values, c].values

final_df.to_parquet(OUTPUT_PATH, index=False)
print(f"Saved → {OUTPUT_PATH}")
print(final_df[CONDITIONS].apply(pd.Series.value_counts).fillna(0).astype(int))