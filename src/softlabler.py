import subprocess
subprocess.run(["pip", "install", "-q", "bitsandbytes>=0.46.1"], check=True)

import time
import numpy as np
import pandas as pd
import kagglehub
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_path = "PUT THE MODEL PATH HERE!!"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto"
)
model.eval()
print("Qwen loaded successfully!")

train_path = "TRAINING CSV FILE"
CONDITIONS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", 
              "Medial OA", "Lateral OA", "PF OA", "Effusion", 
              "Synovitis", "Baker's", "Contusion", "Fracture"]

def extract_labels(report_text):
    prompt = f"""You are a radiologist assistant. Read the MRI knee report below and for each condition respond with:
1 = clearly mentioned/present
0 = clearly denied/absent
0.5 = not mentioned
Conditions to label: {', '.join(CONDITIONS)}
Report: {report_text}
Respond in this exact format, one per line:
ACL: <value>
MCL: <value>
Medial Meniscus: <value>
Lateral Meniscus: <value>
Medial OA: <value>
Lateral OA: <value>
PF OA: <value>
Effusion: <value>
Synovitis: <value>
Baker's: <value>
Contusion: <value>
Fracture: <value>"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False
        )
    response = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], 
                                skip_special_tokens=True).strip()
    labels = {}
    for line in response.split('\n'):
        for cond in CONDITIONS:
            if line.startswith(cond + ':'):
                value = line.split(':')[1].strip()
                try:
                    labels[cond] = float(value)
                except:
                    labels[cond] = 0.5
    for cond in CONDITIONS:
        if cond not in labels:
            labels[cond] = 0.5
    return labels

print("Function ready!")

df = pd.read_csv(train_path)
unlabeled_df = df[df['ACL'].isna()].reset_index(drop=True)
print(f"Total to label: {len(unlabeled_df)}")

checkpoint_path = "CHECKPOINT FILE PATH"
if os.path.exists(checkpoint_path):
    done_df = pd.read_csv(checkpoint_path)
    done_ids = set(done_df['StudyInstanceUID'].tolist())
    print(f"Resuming — already done: {len(done_ids)}")
else:
    done_df = pd.DataFrame()
    done_ids = set()

results = []
for idx, row in unlabeled_df.iterrows():
    if row['StudyInstanceUID'] in done_ids:
        continue
    
    labels = extract_labels(row['Report'])
    labels['StudyInstanceUID'] = row['StudyInstanceUID']
    results.append(labels)
    
    if len(results) % 50 == 0:
        chunk = pd.DataFrame(results)
        done_df = pd.concat([done_df, chunk], ignore_index=True)
        done_df.to_csv(checkpoint_path, index=False)
        results = []
        print(f"Checkpoint saved — total done: {len(done_df)}/{len(unlabeled_df)}")

if results:
    chunk = pd.DataFrame(results)
    done_df = pd.concat([done_df, chunk], ignore_index=True)
    done_df.to_csv(checkpoint_path, index=False)

print(f"Done! Total labeled: {len(done_df)}")
done_df.to_csv("FINAL LABELS PATH", index=False)