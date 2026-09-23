"""Zero-shot multimodal-LLM baseline (protocol after Guzel et al., Diagn Interv Radiol 2026).

Asks each model for a pneumothorax probability (0-100) per image; temperature 0; one pass.
Check each dataset licence before sending images to a third-party API (risk R9).
    OPENAI_API_KEY=... python scripts/mllm_eval.py --provider openai --model gpt-4o --data /workspace/data --set siim:test
Writes results/runs/mllm_<model>/seed0/preds_<ds>_<split>.csv (+ dev-val threshold if --set candid:val is run first).
"""
import argparse
import base64
import json
import math
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

from ptx.data.dataset import load_table
from ptx.eval.metrics import threshold_at_specificity

PROMPT = ("You are assisting with a research evaluation. This is a frontal chest radiograph. "
          "Estimate the probability (0-100) that a pneumothorax is present. "
          "Answer with JSON only: {\"probability\": <number>}")


def ask(provider, model, png_bytes):
    b64 = base64.b64encode(png_bytes).decode()
    if provider == "openai":
        r = requests.post("https://api.openai.com/v1/chat/completions", timeout=120,
                          headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                          json={"model": model, "temperature": 0, "messages": [{"role": "user", "content": [
                              {"type": "text", "text": PROMPT},
                              {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]})
        txt = r.json()["choices"][0]["message"]["content"]
    elif provider == "anthropic":
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=120,
                          headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
                          json={"model": model, "max_tokens": 50, "temperature": 0, "messages": [{"role": "user", "content": [
                              {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                              {"type": "text", "text": PROMPT}]}]})
        txt = r.json()["content"][0]["text"]
    elif provider == "gemini":
        r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                          params={"key": os.environ["GEMINI_API_KEY"]}, timeout=120,
                          json={"generationConfig": {"temperature": 0}, "contents": [{"parts": [
                              {"text": PROMPT}, {"inline_data": {"mime_type": "image/png", "data": b64}}]}]})
        txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        raise ValueError(provider)
    m = re.search(r"(\d+(\.\d+)?)", txt)
    return min(max(float(m.group(1)) / 100 if m else float("nan"), 1e-4), 1 - 1e-4)


ap = argparse.ArgumentParser()
ap.add_argument("--provider", required=True, choices=["openai", "anthropic", "gemini"])
ap.add_argument("--model", required=True)
ap.add_argument("--data", type=Path, required=True)
ap.add_argument("--splits", type=Path, default=Path("splits"))
ap.add_argument("--set", required=True, help="dataset:split")
a = ap.parse_args()
ds, sp = a.set.split(":")
od = Path("results/runs") / f"mllm_{a.model.replace('/', '_')}" / "seed0"
od.mkdir(parents=True, exist_ok=True)
out = od / f"preds_{ds}_{sp}.csv"
done = pd.read_csv(out, dtype={"image_id": str}) if out.exists() else pd.DataFrame(columns=["image_id"])
t = load_table(a.data, ds, a.splits, sp)
rows = done.to_dict("records")
for r in t.itertuples():
    if r.image_id in set(done.image_id):
        continue
    s = ask(a.provider, a.model, (a.data / ds / "img" / f"{r.image_id}.png").read_bytes())
    rows.append({"image_id": r.image_id, "dataset": ds, "patient_id": r.patient_id, "ptx": r.ptx, "treated": r.treated,
                 "treated_source": r.treated_source, "score": s, "logit": math.log(s / (1 - s))})
    if len(rows) % 50 == 0:
        pd.DataFrame(rows).to_csv(out, index=False)
    time.sleep(0.2)
df = pd.DataFrame(rows)
df.to_csv(out, index=False)
if sp == "val":
    json.dump({"threshold_spec95": threshold_at_specificity(df.ptx, df.score, 0.95)}, open(od / "threshold.json", "w"))
    (od / "DONE").write_text("mllm")
print(out, len(df))
